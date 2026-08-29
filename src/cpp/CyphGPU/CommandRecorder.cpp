#include "CommandRecorder.hpp"

#include <CyphGPU/BLAS.hpp>
#include <CyphGPU/ComputePassContext.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/GraphicsPassContext.hpp>
#include <CyphGPU/Queue.hpp>
#include <CyphGPU/TLAS.hpp>

#include <bit>
#include <boost/container/static_vector.hpp>
#include <exception>
#include <ranges>
#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>

#define COMMAND_PARSE \
	ZoneScoped;       \
	assert(!m_submitted);

#define VULKAN_CALL(name) ZoneScopedN(#name);

#define REGIONED_COMMAND_EXECUTE_BEGIN(name)                     \
	TracyVkZone(queue->getTracyContext(), cmd_buf, #name);       \
                                                                 \
	{                                                            \
		vk::DebugUtilsLabelEXT debug_info;                       \
		debug_info.pLabelName = #name;                           \
		debug_info.color = {{0.0f, 0.0f, 0.0f, 0.0f}};           \
                                                                 \
		VULKAN_CALL(vkCmdBeginDebugUtilsLabelEXT);               \
		cmd_buf.beginDebugUtilsLabelEXT(debug_info, dispatcher); \
	}

#define REGIONED_COMMAND_EXECUTE_END               \
	{                                              \
		VULKAN_CALL(vkCmdEndDebugUtilsLabelEXT);   \
		cmd_buf.endDebugUtilsLabelEXT(dispatcher); \
	}

namespace
{
template<class... TOverloads>
struct Overloaded : TOverloads...
{
	using TOverloads::operator()...;
};

[[nodiscard]]
constexpr bool hasWriteAccesses(vk::AccessFlags2 accesses)
{
	vk::AccessFlags2 write_accesses =
		vk::AccessFlagBits2::eShaderWrite |
		vk::AccessFlagBits2::eColorAttachmentWrite |
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
		vk::AccessFlagBits2::eTransferWrite |
		vk::AccessFlagBits2::eHostWrite |
		vk::AccessFlagBits2::eMemoryWrite |
		vk::AccessFlagBits2::eShaderStorageWrite |
		vk::AccessFlagBits2::eVideoDecodeWriteKHR |
		vk::AccessFlagBits2::eVideoEncodeWriteKHR |
		vk::AccessFlagBits2::eShaderTileAttachmentWriteQCOM |
		vk::AccessFlagBits2::eTransformFeedbackWriteEXT |
		vk::AccessFlagBits2::eTransformFeedbackCounterWriteEXT |
		vk::AccessFlagBits2::eCommandPreprocessWriteEXT |
		vk::AccessFlagBits2::eAccelerationStructureWriteKHR |
		vk::AccessFlagBits2::eMicromapWriteEXT |
		vk::AccessFlagBits2::eOpticalFlowWriteNV;

	return static_cast<bool>(accesses & write_accesses);
}

constexpr std::array CLEAR_IMAGE_DEFAULT_RANGE = {
	cgpu::CommandRecorder::ImageLevelsLayersRange{},
};

constexpr std::array COPY_IMAGE_TO_IMAGE_DEFAULT_RANGE = {
	cgpu::CommandRecorder::CopyImageToImageParams::Range{},
};

constexpr std::array COPY_BUFFER_TO_IMAGE_DEFAULT_RANGE = {
	cgpu::CommandRecorder::CopyBufferToImageParams::Range{},
};

constexpr std::array COPY_IMAGE_TO_BUFFER_DEFAULT_RANGE = {
	cgpu::CommandRecorder::CopyImageToBufferParams::Range{},
};

constexpr std::array COPY_BUFFER_TO_BUFFER_DEFAULT_RANGE = {
	cgpu::CommandRecorder::CopyBufferToBufferParams::Range{},
};

constexpr std::array BLIT_DEFAULT_RANGE = {
	cgpu::CommandRecorder::BlitParams::Range{},
};

constexpr std::array RESOLVE_DEFAULT_RANGE = {
	cgpu::CommandRecorder::ResolveParams::Range{},
};

std::tuple<vk::ImageSubresourceRange, vk::DeviceSize> resolveRange(
	const cgpu::ImagePtr& image,
	const cgpu::CommandRecorder::ImageLevelsLayersRange& range,
	vk::ImageAspectFlags aspects
)
{
	vk::ImageSubresourceRange vk_range;
	vk_range.aspectMask = aspects;
	vk_range.baseMipLevel = range.levels ? range.levels->offset : 0;
	vk_range.levelCount = range.levels ? range.levels->size : image->getDesc().levels;
	vk_range.baseArrayLayer = range.layers ? range.layers->offset : 0;
	vk_range.layerCount = range.layers ? range.layers->size : image->getDesc().layers;

	vk::DeviceSize byte_size = cgpu::calcImageByteSize(
		image->getDesc().format,
		image->getDesc().extent,
		{vk_range.baseMipLevel, vk_range.levelCount},
		vk_range.layerCount
	);

	return {vk_range, byte_size};
}

std::tuple<vk::ImageSubresourceLayers, cgpu::Range<glm::uvec3>, vk::DeviceSize> resolveRange(
	const cgpu::ImagePtr& image,
	const cgpu::CommandRecorder::ImageLevelLayersAspectsPixelsRange& range
)
{
	vk::ImageSubresourceLayers vk_range;
	vk_range.aspectMask = range.aspects ? *range.aspects : cgpu::getAspects(image->getDesc().format);
	vk_range.mipLevel = range.level ? *range.level : 0;
	vk_range.baseArrayLayer = range.layers ? range.layers->offset : 0;
	vk_range.layerCount = range.layers ? range.layers->size : image->getDesc().layers;

	cgpu::Range<glm::uvec3> pixel_range =
		range.pixels ?
			*range.pixels :
			cgpu::Range<glm::uvec3>{glm::uvec3{0, 0, 0}, cgpu::calcImageLevelExtent(image->getDesc().extent, vk_range.mipLevel)};

	vk::DeviceSize byte_size = cgpu::calcImageByteSize(
		image->getDesc().format,
		pixel_range.size,
		vk_range.layerCount
	);

	return {vk_range, pixel_range, byte_size};
}

std::tuple<cgpu::Range<vk::DeviceSize>, vk::DeviceSize> resolveRange(
	const cgpu::BufferPtr& buffer,
	const cgpu::CommandRecorder::BufferRange& range
)
{
	cgpu::Range<vk::DeviceSize> vk_range =
		range.byte_range ?
			*range.byte_range :
			cgpu::Range<vk::DeviceSize>{0, buffer->getDesc().size};

	vk::DeviceSize byte_size = vk_range.size;

	return {vk_range, byte_size};
}

std::tuple<vk::ImageSubresourceLayers, glm::uvec3, glm::uvec3, vk::DeviceSize> resolveRange(
	const cgpu::ImagePtr& image,
	const cgpu::CommandRecorder::ImageLevelLayersAspectsRectRange& range
)
{
	vk::ImageSubresourceLayers vk_range;
	vk_range.aspectMask = range.aspects ? *range.aspects : cgpu::getAspects(image->getDesc().format);
	vk_range.mipLevel = range.level ? *range.level : 0;
	vk_range.baseArrayLayer = range.layers ? range.layers->offset : 0;
	vk_range.layerCount = range.layers ? range.layers->size : image->getDesc().layers;

	glm::uvec3 top_left = range.top_left ? *range.top_left : glm::uvec3{0, 0, 0};
	glm::uvec3 bottom_right = range.bottom_right ? *range.bottom_right : cgpu::calcImageLevelExtent(image->getDesc().extent, vk_range.mipLevel);
	glm::uvec3 rect_extent = glm::uvec3{glm::abs(glm::ivec3{bottom_right} - glm::ivec3{top_left})};

	vk::DeviceSize byte_size = cgpu::calcImageByteSize(
		image->getDesc().format,
		rect_extent,
		vk_range.layerCount
	);

	return {vk_range, top_left, bottom_right, byte_size};
}
}

cgpu::CommandRecorder::SubmitHandle cgpu::CommandRecorder::submit()
{
	ZoneScoped;

#if !defined(NDEBUG)
	assert(!m_submitted);
	m_submitted = true;
#endif

	static constexpr uint32_t INVALID_IDX = std::numeric_limits<uint32_t>::max();

	struct LastResourceAccess
	{
		uint32_t cmd_idx{INVALID_IDX};
		uint32_t cmd_stageful_index{};
		AccessPoints access_points{};

		operator bool() const
		{
			return cmd_idx != INVALID_IDX;
		}
	};

	struct LastResourceAccesses
	{
		LastResourceAccess last_write{};
		LastResourceAccess reads_since_last_write{};
	};

	struct Barrier
	{
		uint32_t cmd_idx{};
		Resource* resource{};
		AccessPoints src{};
		AccessPoints dst{};
	};

	struct Event
	{
		vk::Event vk_event{};
		uint32_t vk_image_barrier_begin{0};
		uint32_t vk_image_barrier_end{0};
		uint32_t vk_buffer_barrier_begin{0};
		uint32_t vk_buffer_barrier_end{0};
		vk::StructureChain<vk::DependencyInfo, vk::MemoryRangeBarriersInfoKHR> vk_chain{};
	};

	struct CmdSync
	{
		struct
		{
			bool exists{false};

			uint32_t barrier_begin{0};
			uint32_t barrier_end{0};

			uint32_t event_idx{INVALID_IDX};
		} signal;

		uint32_t dep_begin{0};
		uint32_t dep_end{0};
	};

	struct Dependency
	{
		uint32_t cmd_idx;
		uint32_t dep_cmd_idx;
	};

	detail::BumpDenseUnorderedMap<Resource*, LastResourceAccesses> last_resource_accesses{detail::BumpAllocator{*m_bump_memory}};
	detail::BumpVector<Event> events{detail::BumpAllocator{*m_bump_memory}};
	detail::BumpVector<CmdSync> cmd_syncs{detail::BumpAllocator{*m_bump_memory}};
	detail::BumpVector<Dependency> deps{detail::BumpAllocator{*m_bump_memory}};
	detail::BumpVector<Barrier> barriers{detail::BumpAllocator{*m_bump_memory}};
	{
		ZoneScopedN("Resolve sync");

		last_resource_accesses.reserve(m_num_resources);
		cmd_syncs.resize(m_containers->cmd_list.size());
		deps.reserve(m_containers->cmd_list.size());
		barriers.reserve(m_num_resource_barriers);
		uint32_t cmd_idx = 0;
		uint32_t next_cmd_stageful_idx = 0;
		for (auto& cmd : m_containers->cmd_list)
		{
			uint32_t cmd_stageful_index = next_cmd_stageful_idx;
			if (cmd.is_stageful)
			{
				next_cmd_stageful_idx++;
			}

			cmd_syncs[cmd_idx].dep_begin = static_cast<uint32_t>(deps.size());

			for (const auto& [resource, access_point] : cmd.referenced_resources)
			{
				LastResourceAccesses& global_resource_access = last_resource_accesses[resource];
				LastResourceAccess dep_resource_access{};
				if (!hasWriteAccesses(access_point.accesses))
				{
					if (global_resource_access.last_write) // RaW
					{
						dep_resource_access = global_resource_access.last_write;
					}

					global_resource_access.reads_since_last_write = {
						.cmd_idx = cmd_idx,
						.cmd_stageful_index = cmd_stageful_index,
						.access_points = global_resource_access.reads_since_last_write.access_points | access_point,
					};
				}
				else
				{
					if (global_resource_access.reads_since_last_write) // WaR
					{
						dep_resource_access = global_resource_access.reads_since_last_write;
					}
					else if (global_resource_access.last_write) // WaW
					{
						dep_resource_access = global_resource_access.last_write;
					}

					global_resource_access.reads_since_last_write = {};

					global_resource_access.last_write = {
						.cmd_idx = cmd_idx,
						.cmd_stageful_index = cmd_stageful_index,
						.access_points = access_point,
					};
				}

				if (dep_resource_access)
				{
					auto& dep_cmd_sync = cmd_syncs[dep_resource_access.cmd_idx];
					if (!dep_cmd_sync.signal.exists)
					{
						dep_cmd_sync.signal.exists = true;

						// We have only one wait point per signal point, with the dst masks including
						// all future reads until the next write.
						deps.push_back({
							.cmd_idx = cmd_idx,
							.dep_cmd_idx = dep_resource_access.cmd_idx,
						});

						// If unrelated stageful commands are present between the two, use events instead of barriers
						uint64_t num_stageful_cmds_between = cmd_stageful_index - dep_resource_access.cmd_stageful_index - 1;
						if (num_stageful_cmds_between > 0)
						{
							dep_cmd_sync.signal.event_idx = static_cast<uint32_t>(events.size());

							events.push_back({
								.vk_event = m_slot->createEvent(),
							});
						}
					}

					barriers.push_back({
						.cmd_idx = dep_resource_access.cmd_idx,
						.resource = resource,
						.src = dep_resource_access.access_points,
						.dst = access_point,
					});
				}
			}

			cmd_syncs[cmd_idx].dep_end = static_cast<uint32_t>(deps.size());

			cmd_idx++;
		}

		{
			ZoneScopedN("Sort and dispatch barriers");

			std::ranges::sort(
				barriers,
				[](const auto& a, const auto& b) {
					if (a.cmd_idx != b.cmd_idx)
					{
						return a.cmd_idx < b.cmd_idx;
					}

					return a.resource < b.resource;
				}
			);

			// Merge dst for duplicate barriers
			{
				uint32_t unique_barrier_count = 0;
				for (uint32_t barrier_idx = 0; barrier_idx < barriers.size(); barrier_idx++)
				{
					if (unique_barrier_count > 0 &&
					    barriers[unique_barrier_count - 1].cmd_idx == barriers[barrier_idx].cmd_idx &&
					    barriers[unique_barrier_count - 1].resource == barriers[barrier_idx].resource)
					{
						barriers[unique_barrier_count - 1].dst |= barriers[barrier_idx].dst;
					}
					else
					{
						barriers[unique_barrier_count++] = barriers[barrier_idx];
					}
				}

				barriers.resize(unique_barrier_count);
			}

			// Dispatch barrier offsets/sizes
			{
				uint32_t cur_cmd_idx = INVALID_IDX;
				for (uint32_t barrier_idx = 0; barrier_idx < barriers.size(); barrier_idx++)
				{
					if (barriers[barrier_idx].cmd_idx != cur_cmd_idx)
					{
						cur_cmd_idx = barriers[barrier_idx].cmd_idx;
						cmd_syncs[cur_cmd_idx].signal.barrier_begin = barrier_idx;
						cmd_syncs[cur_cmd_idx].signal.barrier_end = barrier_idx;
					}

					cmd_syncs[cur_cmd_idx].signal.barrier_end++;
				}
			}
		}
	}

	detail::BumpFlatMap<vk::Semaphore, uint64_t> signals_to_wait{detail::BumpAllocator{*m_bump_memory}};
	auto add_signal_to_wait = [&](vk::Semaphore semaphore, uint64_t value) {
		auto [it, inserted] = signals_to_wait.try_emplace(semaphore, value);
		if (!inserted)
		{
			it->second = std::max(it->second, value);
		}
	};

	detail::BumpVector<std::pair<Resource*, bool>> referenced_resources{detail::BumpAllocator{*m_bump_memory}};
	{
		ZoneScopedN("Resource prologue");

		referenced_resources.reserve(last_resource_accesses.size());
		for (auto& [resource, access] : last_resource_accesses)
		{
			referenced_resources.emplace_back(resource, access.last_write);
		}

		// Sort by pointer value to avoid any risk of deadlock with other concurrent submits
		std::ranges::sort(referenced_resources, {}, &std::pair<Resource*, bool>::first);

		for (auto& [resource, written] : referenced_resources)
		{
			resource->lock();

			if (written)
			{
				for (const auto& [semaphore, value] : resource->getReadSignals())
				{
					add_signal_to_wait(semaphore, value);
				}
			}
			else
			{
				const auto& signal = resource->tryGetReadWriteSignal();
				if (signal)
				{
					add_signal_to_wait(signal->semaphore, signal->value);
				}
			}
		}
	}

	vk::CommandBuffer cmd_buf = m_slot->createCommandBuffer(m_queue, vk::CommandBufferLevel::ePrimary);

	{
		vk::CommandBufferBeginInfo info;
		info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
		// info.pInheritanceInfo;

		VULKAN_CALL(vkBeginCommandBuffer);
		cmd_buf.begin(
			info,
			*m_dispatcher
		);
	}

	if (m_queue->getCapabilities() & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute))
	{
		{
			VULKAN_CALL(vkCmdBindResourceHeapEXT);
			cmd_buf.bindResourceHeapEXT(
				m_slot->getDeviceSession()->getResourceBindHeapInfo(),
				*m_dispatcher
			);
		}

		{
			VULKAN_CALL(vkCmdBindSamplerHeapEXT);
			cmd_buf.bindSamplerHeapEXT(
				m_slot->getDeviceSession()->getSamplerBindHeapInfo(),
				*m_dispatcher
			);
		}
	}

	auto barriers_to_vk =
		[&](
			const CmdSync& cmd_sync,
			detail::BumpVector<vk::ImageMemoryBarrier2>& vk_image_barriers,
			detail::BumpVector<vk::MemoryRangeBarrierKHR>& vk_buffer_barriers
		) {
			for (uint32_t barrier_idx = cmd_sync.signal.barrier_begin; barrier_idx != cmd_sync.signal.barrier_end; barrier_idx++)
			{
				const auto& barrier = barriers[barrier_idx];
				if (barrier.resource->getType() == Resource::Type::eImage)
				{
					auto& image = static_cast<Image&>(*barrier.resource);
					vk_image_barriers.push_back({
						.srcStageMask = barrier.src.stages,
						.srcAccessMask = barrier.src.accesses,
						.dstStageMask = barrier.dst.stages,
						.dstAccessMask = barrier.dst.accesses,
						.oldLayout = vk::ImageLayout::eGeneral,
						.newLayout = vk::ImageLayout::eGeneral,
						.srcQueueFamilyIndex = vk::QueueFamilyIgnored,
						.dstQueueFamilyIndex = vk::QueueFamilyIgnored,
						.image = image.getHandle(),
						.subresourceRange = {
							.aspectMask = getAspects(image.getDesc().format),
							.baseMipLevel = 0,
							.levelCount = vk::RemainingMipLevels,
							.baseArrayLayer = 0,
							.layerCount = vk::RemainingArrayLayers,
						},
					});
				}
				else
				{
					auto& buffer = static_cast<Buffer&>(*barrier.resource);
					vk_buffer_barriers.push_back({
						.srcStageMask = barrier.src.stages,
						.srcAccessMask = barrier.src.accesses,
						.dstStageMask = barrier.dst.stages,
						.dstAccessMask = barrier.dst.accesses,
						.srcQueueFamilyIndex = vk::QueueFamilyIgnored,
						.dstQueueFamilyIndex = vk::QueueFamilyIgnored,
						.addressRange = {
							.address = buffer.getDevicePtr(),
							.size = buffer.getDesc().size,
						},
						.addressFlags = vk::AddressCommandFlagBitsKHR::eFullyBound,
					});
				}
			}
		};

	{
		ZoneScopedN("Command recording");

		detail::BumpVector<vk::ImageMemoryBarrier2> event_image_barriers_scratch{detail::BumpAllocator{*m_bump_memory}};
		detail::BumpVector<vk::MemoryRangeBarrierKHR> event_buffer_barriers_scratch{detail::BumpAllocator{*m_bump_memory}};

		detail::BumpVector<vk::Event> wait_events_scratch{detail::BumpAllocator{*m_bump_memory}};
		detail::BumpVector<vk::DependencyInfo> wait_chains_scratch{detail::BumpAllocator{*m_bump_memory}};

		detail::BumpList<vk::Event> reset_events_scratch{detail::BumpAllocator{*m_bump_memory}};

		detail::BumpVector<vk::ImageMemoryBarrier2> vk_images_barriers_scratch{detail::BumpAllocator{*m_bump_memory}};
		detail::BumpVector<vk::MemoryRangeBarrierKHR> vk_buffers_barriers_scratch{detail::BumpAllocator{*m_bump_memory}};
		uint32_t cmd_idx = 0;
		for (auto& cmd : m_containers->cmd_list)
		{
			// Wait events
			if (cmd_syncs[cmd_idx].dep_begin != cmd_syncs[cmd_idx].dep_end)
			{
				for (uint32_t dep_idx = cmd_syncs[cmd_idx].dep_begin; dep_idx != cmd_syncs[cmd_idx].dep_end; dep_idx++)
				{
					uint32_t dep_cmd_idx = deps[dep_idx].dep_cmd_idx;
					auto& dep_cmd_sync = cmd_syncs[dep_cmd_idx];

					if (dep_cmd_sync.signal.event_idx == INVALID_IDX)
					{
						// The signal point was a full barrier instead of an event signal
						continue;
					}

					auto& dep_event = events[dep_cmd_sync.signal.event_idx];

					// Refresh barrier pointers, storage might have been relocated
					auto& dep_info = dep_event.vk_chain.get<vk::DependencyInfo>();
					dep_info.pImageMemoryBarriers = event_image_barriers_scratch.data() + dep_event.vk_image_barrier_begin;

					auto& mem_range_info = dep_event.vk_chain.get<vk::MemoryRangeBarriersInfoKHR>();
					mem_range_info.pMemoryRangeBarriers = event_buffer_barriers_scratch.data() + dep_event.vk_buffer_barrier_begin;

					wait_events_scratch.emplace_back(dep_event.vk_event);
					wait_chains_scratch.emplace_back(dep_event.vk_chain.get());
				}

				if (!wait_events_scratch.empty())
				{
					{
						VULKAN_CALL(vkCmdWaitEvents2);
						cmd_buf.waitEvents2(
							wait_events_scratch,
							wait_chains_scratch,
							*m_dispatcher
						);
					}

					reset_events_scratch.append_range(wait_events_scratch);

					wait_events_scratch.clear();
					wait_chains_scratch.clear();
				}
			}

			// Init image layouts
			{
				for (const auto& [resource, access_point] : cmd.referenced_resources)
				{
					if (resource->getType() != Resource::Type::eImage)
					{
						continue;
					}

					auto& image = static_cast<Image&>(*resource);

					if (image.isLayoutInitialized())
					{
						continue;
					}

					auto& barrier = vk_images_barriers_scratch.emplace_back();
					barrier.srcStageMask = vk::PipelineStageFlagBits2::eNone;
					barrier.srcAccessMask = vk::AccessFlagBits2::eNone;
					barrier.dstStageMask = access_point.stages;
					barrier.dstAccessMask = access_point.accesses;
					barrier.oldLayout = vk::ImageLayout::eUndefined;
					barrier.newLayout = vk::ImageLayout::eGeneral;
					barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
					barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
					barrier.image = image.getHandle();
					barrier.subresourceRange.aspectMask = getAspects(image.getDesc().format);
					barrier.subresourceRange.baseMipLevel = 0;
					barrier.subresourceRange.levelCount = vk::RemainingMipLevels;
					barrier.subresourceRange.baseArrayLayer = 0;
					barrier.subresourceRange.layerCount = vk::RemainingArrayLayers;

					image.setLayoutInitialized(true);
				}

				if (!vk_images_barriers_scratch.empty())
				{
					vk::DependencyInfo dep_info;
					dep_info.dependencyFlags = {};
					dep_info.memoryBarrierCount = 0;
					// dep_info.pMemoryBarriers;
					dep_info.bufferMemoryBarrierCount = 0;
					// dep_info.pBufferMemoryBarriers;
					dep_info.imageMemoryBarrierCount = static_cast<uint32_t>(vk_images_barriers_scratch.size());
					dep_info.pImageMemoryBarriers = vk_images_barriers_scratch.data();

					{
						VULKAN_CALL(vkCmdPipelineBarrier2);
						cmd_buf.pipelineBarrier2(
							dep_info,
							*m_dispatcher
						);
					}

					vk_images_barriers_scratch.clear();
				}
			}

			// Execute command
			(*cmd.callback)(m_queue, cmd_buf, *m_dispatcher);

			// Signal event
			const auto& cmd_sync = cmd_syncs[cmd_idx];
			if (cmd_sync.signal.exists)
			{
				if (cmd_sync.signal.event_idx != INVALID_IDX)
				{
					auto& event = events[cmd_sync.signal.event_idx];

					event.vk_image_barrier_begin = static_cast<uint32_t>(event_image_barriers_scratch.size());
					event.vk_buffer_barrier_begin = static_cast<uint32_t>(event_buffer_barriers_scratch.size());
					barriers_to_vk(
						cmd_sync,
						event_image_barriers_scratch,
						event_buffer_barriers_scratch
					);
					event.vk_image_barrier_end = static_cast<uint32_t>(event_image_barriers_scratch.size());
					event.vk_buffer_barrier_end = static_cast<uint32_t>(event_buffer_barriers_scratch.size());

					auto& dep_info = event.vk_chain.get<vk::DependencyInfo>();
					dep_info.dependencyFlags = {};
					dep_info.memoryBarrierCount = 0;
					// dep_info.pMemoryBarriers;
					dep_info.bufferMemoryBarrierCount = 0;
					// dep_info.pBufferMemoryBarriers;
					dep_info.imageMemoryBarrierCount = event.vk_image_barrier_end - event.vk_image_barrier_begin;
					dep_info.pImageMemoryBarriers = event_image_barriers_scratch.data() + event.vk_image_barrier_begin;

					auto& mem_range_info = event.vk_chain.get<vk::MemoryRangeBarriersInfoKHR>();
					mem_range_info.memoryRangeBarrierCount = event.vk_buffer_barrier_end - event.vk_buffer_barrier_begin;
					mem_range_info.pMemoryRangeBarriers = event_buffer_barriers_scratch.data() + event.vk_buffer_barrier_begin;

					{
						VULKAN_CALL(vkCmdSetEvent2);
						cmd_buf.setEvent2(
							event.vk_event,
							event.vk_chain.get(),
							*m_dispatcher
						);
					}
				}
				else
				{
					vk::StructureChain<vk::DependencyInfo, vk::MemoryRangeBarriersInfoKHR> vk_chain;
					barriers_to_vk(
						cmd_sync,
						vk_images_barriers_scratch,
						vk_buffers_barriers_scratch
					);

					auto& dep_info = vk_chain.get<vk::DependencyInfo>();
					dep_info.dependencyFlags = {};
					dep_info.memoryBarrierCount = 0;
					// dep_info.pMemoryBarriers;
					dep_info.bufferMemoryBarrierCount = 0;
					// dep_info.pBufferMemoryBarriers;
					dep_info.imageMemoryBarrierCount = static_cast<uint32_t>(vk_images_barriers_scratch.size());
					dep_info.pImageMemoryBarriers = vk_images_barriers_scratch.data();

					auto& mem_range_info = vk_chain.get<vk::MemoryRangeBarriersInfoKHR>();
					mem_range_info.memoryRangeBarrierCount = static_cast<uint32_t>(vk_buffers_barriers_scratch.size());
					mem_range_info.pMemoryRangeBarriers = vk_buffers_barriers_scratch.data();

					{
						VULKAN_CALL(vkCmdPipelineBarrier2);
						cmd_buf.pipelineBarrier2(
							vk_chain.get(),
							*m_dispatcher
						);
					}

					vk_images_barriers_scratch.clear();
					vk_buffers_barriers_scratch.clear();
				}
			}

			cmd_idx++;
		}

		// We can't reset events right after their wait because some drivers don't respect the
		// absence of a second sync scope in vkCmdResetEvent2 and would stall the next command
		for (auto event : reset_events_scratch)
		{
			VULKAN_CALL(vkCmdResetEvent2);
			cmd_buf.resetEvent2(
				event,
				vk::PipelineStageFlagBits2::eAllCommands,
				*m_dispatcher
			);
		}
	}

	{
		VULKAN_CALL(vkEndCommandBuffer);
		cmd_buf.end(
			*m_dispatcher
		);
	}

	std::vector<std::shared_ptr<void>> referenced_objects;
	referenced_objects.reserve(m_containers->referenced_objects.size());
	for (auto& object : m_containers->referenced_objects | std::views::keys)
	{
		referenced_objects.emplace_back(std::move(object));
	}

	m_containers->referenced_objects.clear();

	Queue::Signal signal = m_queue->submit(
		*m_bump_memory,
		{{cmd_buf}},
		signals_to_wait.keys(),
		signals_to_wait.values(),
		std::move(referenced_objects)
	);

	{
		ZoneScopedN("Resource epilogue");

		for (auto& [resource, written] : referenced_resources)
		{
			written ?
				resource->setReadWriteSignal(signal) :
				resource->addReadSignal(signal);

			resource->unlock();
		}
	}

	m_slot->addFinishedSignal(signal);

	m_containers.reset();

	return {m_slot->getDeviceSession(), signal};
}

void cgpu::CommandRecorder::SubmitHandle::waitFinished() const
{
	std::ignore = waitSemaphore(std::numeric_limits<uint64_t>::max());
}

bool cgpu::CommandRecorder::SubmitHandle::isFinished() const
{
	return waitSemaphore(0) == vk::Result::eSuccess;
}

cgpu::CommandRecorder::SubmitHandle::SubmitHandle(const DeviceSessionPtr& device_session, Queue::Signal signal):
	m_device_session{device_session},
	m_signal{signal}
{
}

vk::Result cgpu::CommandRecorder::SubmitHandle::waitSemaphore(uint64_t timeout) const
{
	vk::SemaphoreWaitInfo info;
	info.flags = {};
	info.semaphoreCount = 1;
	info.pSemaphores = &m_signal.semaphore;
	info.pValues = &m_signal.value;

	return m_device_session->getHandle().waitSemaphores(info, timeout, m_device_session->getDispatcher());
}

// NOLINTBEGIN(*-rvalue-reference-param-not-moved)

void cgpu::CommandRecorder::clearImage(ClearImageParams&& params)
{
	COMMAND_PARSE

	vk::ImageAspectFlags aspects;
	if (params.color_value)
	{
		aspects |= vk::ImageAspectFlagBits::eColor;
	}
	if (params.depth_value)
	{
		aspects |= vk::ImageAspectFlagBits::eDepth;
	}
	if (params.stencil_value)
	{
		aspects |= vk::ImageAspectFlagBits::eStencil;
	}

	if (!aspects)
	{
		return;
	}

	std::span<const ImageLevelsLayersRange> ranges = params.ranges ? std::span{std::as_const(*params.ranges)} : CLEAR_IMAGE_DEFAULT_RANGE;
	detail::BumpVector<vk::ImageSubresourceRange> vk_ranges{detail::BumpAllocator{*m_bump_memory}};
	vk_ranges.reserve(ranges.size());
	for (const auto& range : ranges)
	{
		auto [vk_range, byte_size] = resolveRange(*params.image, range, aspects);

		if (byte_size == 0)
		{
			continue;
		}

		vk_ranges.emplace_back(vk_range);
	}

	if (vk_ranges.empty())
	{
		return;
	}

	struct CmdCallback final : CmdCallbackBase
	{
		detail::BumpVector<vk::ImageSubresourceRange> ranges;
		vk::Image image;
		std::optional<vk::ClearColorValue> color_value;
		std::optional<vk::ClearDepthStencilValue> depth_stencil_value;

		explicit CmdCallback(detail::BumpVector<vk::ImageSubresourceRange>&& ranges):
			ranges{std::move(ranges)}
		{}

		void operator()([[maybe_unused]] const QueuePtr& queue, vk::CommandBuffer cmd_buf, const vk::detail::DispatchLoaderDynamic& dispatcher) override
		{
			REGIONED_COMMAND_EXECUTE_BEGIN(clearImage)

			if (color_value)
			{
				VULKAN_CALL(vkCmdClearColorImage);
				cmd_buf.clearColorImage(
					image,
					vk::ImageLayout::eGeneral,
					*color_value,
					ranges,
					dispatcher
				);
			}

			if (depth_stencil_value)
			{
				VULKAN_CALL(vkCmdClearDepthStencilImage);
				cmd_buf.clearDepthStencilImage(
					image,
					vk::ImageLayout::eGeneral,
					*depth_stencil_value,
					ranges,
					dispatcher
				);
			}

			REGIONED_COMMAND_EXECUTE_END
		}
	};

	auto& cmd = addCmd<CmdCallback>(true, std::move(vk_ranges));
	cmd.image = (*params.image)->getHandle();
	if (params.color_value)
	{
		cmd.color_value = std::visit(
			Overloaded{
				[&](const glm::vec4& value) { return vk::ClearColorValue{.float32 = {{value.r, value.g, value.b, value.a}}}; },
				[&](const glm::ivec4& value) { return vk::ClearColorValue{.int32 = {{value.r, value.g, value.b, value.a}}}; },
				[&](const glm::uvec4& value) { return vk::ClearColorValue{.uint32 = {{value.r, value.g, value.b, value.a}}}; },
			},
			*params.color_value
		);
	}
	if (params.depth_value || params.stencil_value)
	{
		cmd.depth_stencil_value = {{
			params.depth_value.value_or(0.0f),
			params.stencil_value.value_or(0),
		}};
	}

	addCmdResource(
		*params.image,
		{
			vk::PipelineStageFlagBits2::eClear,
			vk::AccessFlagBits2::eTransferWrite,
		}
	);
}

void cgpu::CommandRecorder::copyImageToImage(CopyImageToImageParams&& params)
{
	COMMAND_PARSE

	std::span<const CopyImageToImageParams::Range> ranges = params.ranges ? std::span{std::as_const(*params.ranges)} : COPY_IMAGE_TO_IMAGE_DEFAULT_RANGE;
	detail::BumpVector<vk::ImageCopy2> vk_regions{detail::BumpAllocator{*m_bump_memory}};
	vk_regions.reserve(ranges.size());
	for (const auto& range : ranges)
	{
		auto [src_vk_range, src_pixel_range, src_byte_size] = resolveRange(*params.src_image, range.src.value_or(ImageLevelLayersAspectsPixelsRange{}));
		auto [dst_vk_range, dst_pixel_range, dst_byte_size] = resolveRange(*params.dst_image, range.dst.value_or(ImageLevelLayersAspectsPixelsRange{}));

		if (src_vk_range.layerCount != dst_vk_range.layerCount)
		{
			throw std::logic_error("Image ranges must have the same number of layers.");
		}

		if (src_pixel_range.size != dst_pixel_range.size)
		{
			throw std::logic_error("Image ranges must have the same pixel region size.");
		}

		if (src_byte_size != dst_byte_size)
		{
			throw std::logic_error("Image ranges must have the same byte size.");
		}

		if (dst_byte_size == 0)
		{
			continue;
		}

		vk::ImageCopy2& vk_region = vk_regions.emplace_back();
		vk_region.srcSubresource = src_vk_range;
		vk_region.srcOffset.x = static_cast<int>(src_pixel_range.offset.x);
		vk_region.srcOffset.y = static_cast<int>(src_pixel_range.offset.y);
		vk_region.srcOffset.z = static_cast<int>(src_pixel_range.offset.z);
		vk_region.dstSubresource = dst_vk_range;
		vk_region.dstOffset.x = static_cast<int>(dst_pixel_range.offset.x);
		vk_region.dstOffset.y = static_cast<int>(dst_pixel_range.offset.y);
		vk_region.dstOffset.z = static_cast<int>(dst_pixel_range.offset.z);
		vk_region.extent.width = src_pixel_range.size.x;
		vk_region.extent.height = src_pixel_range.size.y;
		vk_region.extent.depth = src_pixel_range.size.z;
	}

	if (vk_regions.empty())
	{
		return;
	}

	struct CmdCallback final : CmdCallbackBase
	{
		detail::BumpVector<vk::ImageCopy2> regions;
		vk::CopyImageInfo2 info;

		explicit CmdCallback(detail::BumpVector<vk::ImageCopy2>&& regions):
			regions{std::move(regions)}
		{}

		void operator()([[maybe_unused]] const QueuePtr& queue, vk::CommandBuffer cmd_buf, const vk::detail::DispatchLoaderDynamic& dispatcher) override
		{
			REGIONED_COMMAND_EXECUTE_BEGIN(copyImageToImage)

			{
				VULKAN_CALL(vkCmdCopyImage2);
				cmd_buf.copyImage2(
					info,
					dispatcher
				);
			}

			REGIONED_COMMAND_EXECUTE_END
		}
	};

	auto& cmd = addCmd<CmdCallback>(true, std::move(vk_regions));
	cmd.info.srcImage = (*params.src_image)->getHandle();
	cmd.info.srcImageLayout = vk::ImageLayout::eGeneral;
	cmd.info.dstImage = (*params.dst_image)->getHandle();
	cmd.info.dstImageLayout = vk::ImageLayout::eGeneral;
	cmd.info.regionCount = static_cast<uint32_t>(cmd.regions.size());
	cmd.info.pRegions = cmd.regions.data();

	addCmdResource(
		*params.src_image,
		{
			vk::PipelineStageFlagBits2::eCopy,
			vk::AccessFlagBits2::eTransferRead,
		}
	);

	addCmdResource(
		*params.dst_image,
		{
			vk::PipelineStageFlagBits2::eCopy,
			vk::AccessFlagBits2::eTransferWrite,
		}
	);
}

void cgpu::CommandRecorder::copyBufferToImage(CopyBufferToImageParams&& params)
{
	COMMAND_PARSE

	std::span<const CopyBufferToImageParams::Range> ranges = params.ranges ? std::span{std::as_const(*params.ranges)} : COPY_BUFFER_TO_IMAGE_DEFAULT_RANGE;
	detail::BumpVector<vk::DeviceMemoryImageCopyKHR> vk_regions{detail::BumpAllocator{*m_bump_memory}};
	vk_regions.reserve(ranges.size());
	for (const auto& range : ranges)
	{
		auto [src_vk_range, src_byte_size] = resolveRange(*params.src_buffer, range.src.value_or(BufferRange{}));
		auto [dst_vk_range, dst_pixel_range, dst_byte_size] = resolveRange(*params.dst_image, range.dst.value_or(ImageLevelLayersAspectsPixelsRange{}));

		if (src_byte_size != dst_byte_size)
		{
			throw std::logic_error("Image range and buffer range must have the same byte size.");
		}

		if (dst_byte_size == 0)
		{
			continue;
		}

		vk::DeviceMemoryImageCopyKHR& vk_region = vk_regions.emplace_back();
		vk_region.addressRange.address = (*params.src_buffer)->getDevicePtr() + src_vk_range.offset;
		vk_region.addressRange.size = src_vk_range.size;
		vk_region.addressFlags = vk::AddressCommandFlagBitsKHR::eFullyBound;
		vk_region.addressRowLength = 0;
		vk_region.addressImageHeight = 0;
		vk_region.imageLayout = vk::ImageLayout::eGeneral;
		vk_region.imageSubresource = dst_vk_range;
		vk_region.imageOffset.x = static_cast<int>(dst_pixel_range.offset.x);
		vk_region.imageOffset.y = static_cast<int>(dst_pixel_range.offset.y);
		vk_region.imageOffset.z = static_cast<int>(dst_pixel_range.offset.z);
		vk_region.imageExtent.width = dst_pixel_range.size.x;
		vk_region.imageExtent.height = dst_pixel_range.size.y;
		vk_region.imageExtent.depth = dst_pixel_range.size.z;
	}

	if (vk_regions.empty())
	{
		return;
	}

	struct CmdCallback final : CmdCallbackBase
	{
		detail::BumpVector<vk::DeviceMemoryImageCopyKHR> regions;
		vk::CopyDeviceMemoryImageInfoKHR info;

		explicit CmdCallback(detail::BumpVector<vk::DeviceMemoryImageCopyKHR>&& regions):
			regions{std::move(regions)}
		{}

		void operator()([[maybe_unused]] const QueuePtr& queue, vk::CommandBuffer cmd_buf, const vk::detail::DispatchLoaderDynamic& dispatcher) override
		{
			REGIONED_COMMAND_EXECUTE_BEGIN(copyBufferToImage)

			{
				VULKAN_CALL(vkCmdCopyMemoryToImageKHR);
				cmd_buf.copyMemoryToImageKHR(
					info,
					dispatcher
				);
			}

			REGIONED_COMMAND_EXECUTE_END
		}
	};

	auto& cmd = addCmd<CmdCallback>(true, std::move(vk_regions));
	cmd.info.image = (*params.dst_image)->getHandle();
	cmd.info.regionCount = static_cast<uint32_t>(cmd.regions.size());
	cmd.info.pRegions = cmd.regions.data();

	addCmdResource(
		*params.src_buffer,
		{
			vk::PipelineStageFlagBits2::eCopy,
			vk::AccessFlagBits2::eTransferRead,
		}
	);
	addCmdResource(
		*params.dst_image,
		{
			vk::PipelineStageFlagBits2::eCopy,
			vk::AccessFlagBits2::eTransferWrite,
		}
	);
}

void cgpu::CommandRecorder::copyImageToBuffer(CopyImageToBufferParams&& params)
{
	COMMAND_PARSE

	std::span<const CopyImageToBufferParams::Range> ranges = params.ranges ? std::span{std::as_const(*params.ranges)} : COPY_IMAGE_TO_BUFFER_DEFAULT_RANGE;
	detail::BumpVector<vk::DeviceMemoryImageCopyKHR> vk_regions{detail::BumpAllocator{*m_bump_memory}};
	vk_regions.reserve(ranges.size());
	for (const auto& range : ranges)
	{
		auto [src_vk_range, src_pixel_range, src_byte_size] = resolveRange(*params.src_image, range.src.value_or(ImageLevelLayersAspectsPixelsRange{}));
		auto [dst_vk_range, dst_byte_size] = resolveRange(*params.dst_buffer, range.dst.value_or(BufferRange{}));

		if (src_byte_size != dst_byte_size)
		{
			throw std::logic_error("Image range and buffer range must have the same byte size.");
		}

		if (dst_byte_size == 0)
		{
			continue;
		}

		vk::DeviceMemoryImageCopyKHR& vk_region = vk_regions.emplace_back();
		vk_region.addressRange.address = (*params.dst_buffer)->getDevicePtr() + dst_vk_range.offset;
		vk_region.addressRange.size = dst_vk_range.size;
		vk_region.addressFlags = vk::AddressCommandFlagBitsKHR::eFullyBound;
		vk_region.addressRowLength = 0;
		vk_region.addressImageHeight = 0;
		vk_region.imageSubresource = src_vk_range;
		vk_region.imageLayout = vk::ImageLayout::eGeneral;
		vk_region.imageOffset.x = static_cast<int>(src_pixel_range.offset.x);
		vk_region.imageOffset.y = static_cast<int>(src_pixel_range.offset.y);
		vk_region.imageOffset.z = static_cast<int>(src_pixel_range.offset.z);
		vk_region.imageExtent.width = src_pixel_range.size.x;
		vk_region.imageExtent.height = src_pixel_range.size.y;
		vk_region.imageExtent.depth = src_pixel_range.size.z;
	}

	if (vk_regions.empty())
	{
		return;
	}

	struct CmdCallback final : CmdCallbackBase
	{
		detail::BumpVector<vk::DeviceMemoryImageCopyKHR> regions;
		vk::CopyDeviceMemoryImageInfoKHR info;

		explicit CmdCallback(detail::BumpVector<vk::DeviceMemoryImageCopyKHR>&& regions):
			regions{std::move(regions)}
		{}

		void operator()([[maybe_unused]] const QueuePtr& queue, vk::CommandBuffer cmd_buf, const vk::detail::DispatchLoaderDynamic& dispatcher) override
		{
			REGIONED_COMMAND_EXECUTE_BEGIN(copyImageToBuffer)

			{
				VULKAN_CALL(vkCmdCopyImageToMemoryKHR);
				cmd_buf.copyImageToMemoryKHR(
					info,
					dispatcher
				);
			}

			REGIONED_COMMAND_EXECUTE_END
		}
	};

	auto& cmd = addCmd<CmdCallback>(true, std::move(vk_regions));
	cmd.info.image = (*params.src_image)->getHandle();
	cmd.info.regionCount = static_cast<uint32_t>(cmd.regions.size());
	cmd.info.pRegions = cmd.regions.data();

	addCmdResource(
		*params.src_image,
		{
			vk::PipelineStageFlagBits2::eCopy,
			vk::AccessFlagBits2::eTransferRead,
		}
	);
	addCmdResource(
		*params.dst_buffer,
		{
			vk::PipelineStageFlagBits2::eCopy,
			vk::AccessFlagBits2::eTransferWrite,
		}
	);
}

void cgpu::CommandRecorder::copyBufferToBuffer(CopyBufferToBufferParams&& params)
{
	COMMAND_PARSE

	std::span<const CopyBufferToBufferParams::Range> ranges = params.ranges ? std::span{std::as_const(*params.ranges)} : COPY_BUFFER_TO_BUFFER_DEFAULT_RANGE;
	detail::BumpVector<vk::DeviceMemoryCopyKHR> vk_regions{detail::BumpAllocator{*m_bump_memory}};
	vk_regions.reserve(ranges.size());
	for (const auto& range : ranges)
	{
		auto [src_vk_range, src_byte_size] = resolveRange(*params.src_buffer, range.src.value_or(BufferRange{}));
		auto [dst_vk_range, dst_byte_size] = resolveRange(*params.dst_buffer, range.dst.value_or(BufferRange{}));

		if (src_byte_size != dst_byte_size)
		{
			throw std::logic_error("Buffer ranges must have the same byte size.");
		}

		if (dst_byte_size == 0)
		{
			continue;
		}

		vk::DeviceMemoryCopyKHR& vk_region = vk_regions.emplace_back();
		vk_region.srcRange.address = (*params.src_buffer)->getDevicePtr() + src_vk_range.offset;
		vk_region.srcRange.size = src_vk_range.size;
		vk_region.srcFlags = vk::AddressCommandFlagBitsKHR::eFullyBound;
		vk_region.dstRange.address = (*params.dst_buffer)->getDevicePtr() + dst_vk_range.offset;
		vk_region.dstRange.size = dst_vk_range.size;
		vk_region.dstFlags = vk::AddressCommandFlagBitsKHR::eFullyBound;
	}

	if (vk_regions.empty())
	{
		return;
	}

	struct CmdCallback final : CmdCallbackBase
	{
		detail::BumpVector<vk::DeviceMemoryCopyKHR> regions;
		vk::CopyDeviceMemoryInfoKHR info;

		explicit CmdCallback(detail::BumpVector<vk::DeviceMemoryCopyKHR>&& regions):
			regions{std::move(regions)}
		{}

		void operator()([[maybe_unused]] const QueuePtr& queue, vk::CommandBuffer cmd_buf, const vk::detail::DispatchLoaderDynamic& dispatcher) override
		{
			REGIONED_COMMAND_EXECUTE_BEGIN(copyBufferToBuffer)

			{
				VULKAN_CALL(vkCmdCopyMemoryKHR);
				cmd_buf.copyMemoryKHR(
					info,
					dispatcher
				);
			}

			REGIONED_COMMAND_EXECUTE_END
		}
	};

	auto& cmd = addCmd<CmdCallback>(true, std::move(vk_regions));
	cmd.info.regionCount = static_cast<uint32_t>(cmd.regions.size());
	cmd.info.pRegions = cmd.regions.data();

	addCmdResource(
		*params.src_buffer,
		{
			vk::PipelineStageFlagBits2::eCopy,
			vk::AccessFlagBits2::eTransferRead,
		}
	);
	addCmdResource(
		*params.dst_buffer,
		{
			vk::PipelineStageFlagBits2::eCopy,
			vk::AccessFlagBits2::eTransferWrite,
		}
	);
}

void cgpu::CommandRecorder::blit(BlitParams&& params)
{
	COMMAND_PARSE

	std::span<const BlitParams::Range> ranges = params.ranges ? std::span{std::as_const(*params.ranges)} : BLIT_DEFAULT_RANGE;
	detail::BumpVector<vk::ImageBlit2> vk_regions{detail::BumpAllocator{*m_bump_memory}};
	vk_regions.reserve(ranges.size());
	for (const auto& range : ranges)
	{
		auto [src_vk_range, src_top_left, src_bottom_right, src_byte_size] = resolveRange(*params.src_image, range.src.value_or(ImageLevelLayersAspectsRectRange{}));
		auto [dst_vk_range, dst_top_left, dst_bottom_right, dst_byte_size] = resolveRange(*params.dst_image, range.dst.value_or(ImageLevelLayersAspectsRectRange{}));

		if (src_vk_range.layerCount != dst_vk_range.layerCount)
		{
			throw std::logic_error("Image ranges must have the same number of layers.");
		}

		if (dst_byte_size == 0)
		{
			continue;
		}

		vk::ImageBlit2& vk_region = vk_regions.emplace_back();
		vk_region.srcSubresource = src_vk_range;
		vk_region.srcOffsets[0].x = static_cast<int>(src_top_left.x);
		vk_region.srcOffsets[0].y = static_cast<int>(src_top_left.y);
		vk_region.srcOffsets[0].z = static_cast<int>(src_top_left.z);
		vk_region.srcOffsets[1].x = static_cast<int>(src_bottom_right.x);
		vk_region.srcOffsets[1].y = static_cast<int>(src_bottom_right.y);
		vk_region.srcOffsets[1].z = static_cast<int>(src_bottom_right.z);
		vk_region.dstSubresource = dst_vk_range;
		vk_region.dstOffsets[0].x = static_cast<int>(dst_top_left.x);
		vk_region.dstOffsets[0].y = static_cast<int>(dst_top_left.y);
		vk_region.dstOffsets[0].z = static_cast<int>(dst_top_left.z);
		vk_region.dstOffsets[1].x = static_cast<int>(dst_bottom_right.x);
		vk_region.dstOffsets[1].y = static_cast<int>(dst_bottom_right.y);
		vk_region.dstOffsets[1].z = static_cast<int>(dst_bottom_right.z);
	}

	if (vk_regions.empty())
	{
		return;
	}

	struct CmdCallback final : CmdCallbackBase
	{
		detail::BumpVector<vk::ImageBlit2> regions;
		vk::BlitImageInfo2 info;

		explicit CmdCallback(detail::BumpVector<vk::ImageBlit2>&& regions):
			regions{std::move(regions)}
		{}

		void operator()([[maybe_unused]] const QueuePtr& queue, vk::CommandBuffer cmd_buf, const vk::detail::DispatchLoaderDynamic& dispatcher) override
		{
			REGIONED_COMMAND_EXECUTE_BEGIN(blit)

			{
				VULKAN_CALL(vkCmdBlitImage2);
				cmd_buf.blitImage2(
					info,
					dispatcher
				);
			}

			REGIONED_COMMAND_EXECUTE_END
		}
	};

	auto& cmd = addCmd<CmdCallback>(true, std::move(vk_regions));
	cmd.info.srcImage = (*params.src_image)->getHandle();
	cmd.info.srcImageLayout = vk::ImageLayout::eGeneral;
	cmd.info.dstImage = (*params.dst_image)->getHandle();
	cmd.info.dstImageLayout = vk::ImageLayout::eGeneral;
	cmd.info.regionCount = static_cast<uint32_t>(cmd.regions.size());
	cmd.info.pRegions = cmd.regions.data();
	cmd.info.filter = params.filter.value_or(vk::Filter::eNearest);

	addCmdResource(
		*params.src_image,
		{
			vk::PipelineStageFlagBits2::eBlit,
			vk::AccessFlagBits2::eTransferRead,
		}
	);
	addCmdResource(
		*params.dst_image,
		{
			vk::PipelineStageFlagBits2::eBlit,
			vk::AccessFlagBits2::eTransferWrite,
		}
	);
}

void cgpu::CommandRecorder::graphicsPass(GraphicsPassParams&& params)
{
	COMMAND_PARSE

	uint32_t layer_count = 1;
	uint32_t view_mask = 0;
	if (params.layer_mode)
	{
		std::visit(
			Overloaded{
				[&](const GraphicsPassParams::LayerCount& value) {layer_count = *value.value; view_mask = 0; },
				[&](const GraphicsPassParams::MultiviewMask& value) {view_mask = *value.value; layer_count = 0; },
			},
			*params.layer_mode
		);
	}

	uint32_t view_layer_count = layer_count > 0 ? layer_count : std::bit_width(view_mask);
	if (view_layer_count == 0)
	{
		return;
	}

	std::optional<glm::uvec2> implicit_extent;
	bool different_extents = false;
	auto handle_implicit_extent = [&](const ImagePtr& image, uint32_t level) {
		glm::uvec2 extent = calcImageLevelExtent(image->getDesc().extent, level);
		if (!implicit_extent)
		{
			implicit_extent = extent;
		}
		else if (implicit_extent != extent)
		{
			different_extents = true;
		}
	};

	auto color_attachments = params.color_attachments ? *params.color_attachments : std::span<const GraphicsPassParams::ColorAttachment>{};
	boost::container::static_vector<vk::RenderingAttachmentInfo, 8> vk_color_attachments{};
	for (const auto& attachment : color_attachments)
	{
		uint32_t level = attachment.level ? *attachment.level : 0;

		handle_implicit_extent(*attachment.image, level);

		vk::Format format = attachment.format ? *attachment.format : (*attachment.image)->getDesc().format;

		uint32_t first_layer = attachment.first_layer ? *attachment.first_layer : 0;

		// clang-format off
		vk::ImageView view = (*attachment.image)->getAttachmentView(
			format,
			level,
			{first_layer, view_layer_count},
			vk::ImageAspectFlagBits::eColor,
			vk::ImageUsageFlagBits::eColorAttachment
		);
		// clang-format on

		vk::ResolveModeFlagBits resolve_mode = vk::ResolveModeFlagBits::eNone;
		vk::ImageView resolve_view = nullptr;
		if (attachment.resolve)
		{
			resolve_mode =
				attachment.resolve->mode ?
					*attachment.resolve->mode :
					vk::ResolveModeFlagBits::eAverage;

			uint32_t resolve_level = attachment.resolve->level ? *attachment.resolve->level : 0;

			handle_implicit_extent(*attachment.resolve->image, resolve_level);

			uint32_t resolve_first_layer = attachment.resolve->first_layer ? *attachment.resolve->first_layer : 0;

			// clang-format off
			resolve_view = (*attachment.resolve->image)->getAttachmentView(
				format,
				resolve_level,
				{resolve_first_layer, view_layer_count},
				vk::ImageAspectFlagBits::eColor,
				vk::ImageUsageFlagBits::eColorAttachment
			);
			// clang-format on
		}

		vk::ClearColorValue clear_value{};
		if (*attachment.load_op == vk::AttachmentLoadOp::eClear)
		{
			clear_value = std::visit(
				Overloaded{
					[&](const glm::vec4& value) { return vk::ClearColorValue{.float32 = {{value.r, value.g, value.b, value.a}}}; },
					[&](const glm::ivec4& value) { return vk::ClearColorValue{.int32 = {{value.r, value.g, value.b, value.a}}}; },
					[&](const glm::uvec4& value) { return vk::ClearColorValue{.uint32 = {{value.r, value.g, value.b, value.a}}}; },
				},
				attachment.clear_color_value.value()
			);
		}

		auto& vk_attachment = vk_color_attachments.emplace_back();
		vk_attachment.imageView = view;
		vk_attachment.imageLayout = vk::ImageLayout::eGeneral;
		vk_attachment.resolveMode = resolve_mode;
		vk_attachment.resolveImageView = resolve_view;
		vk_attachment.resolveImageLayout = vk::ImageLayout::eGeneral;
		vk_attachment.loadOp = *attachment.load_op;
		vk_attachment.storeOp = *attachment.store_op;
		vk_attachment.clearValue.color = clear_value;
	}

	std::optional<vk::RenderingAttachmentInfo> vk_depth_attachment;
	std::optional<vk::RenderingAttachmentInfo> vk_stencil_attachment;
	if (params.depth_stencil_attachment)
	{
		auto aspects = getAspects((*params.depth_stencil_attachment->image)->getDesc().format);
		bool aspects_have_depth = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		bool aspects_have_stencil = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);

		bool enable_depth = params.depth_stencil_attachment->enable_depth ? *params.depth_stencil_attachment->enable_depth : aspects_have_depth;
		bool enable_stencil = params.depth_stencil_attachment->enable_stencil ? *params.depth_stencil_attachment->enable_stencil : aspects_have_stencil;

		if (enable_depth || enable_stencil)
		{
			uint32_t level = params.depth_stencil_attachment->level ? *params.depth_stencil_attachment->level : 0;

			handle_implicit_extent(*params.depth_stencil_attachment->image, level);

			uint32_t first_layer = params.depth_stencil_attachment->first_layer ? *params.depth_stencil_attachment->first_layer : 0;

			vk::ImageAspectFlags actual_aspects;
			if (enable_depth)
			{
				actual_aspects |= vk::ImageAspectFlagBits::eDepth;
			}
			if (enable_stencil)
			{
				actual_aspects |= vk::ImageAspectFlagBits::eStencil;
			}

			// clang-format off
			vk::ImageView view = (*params.depth_stencil_attachment->image)->getAttachmentView(
				(*params.depth_stencil_attachment->image)->getDesc().format,
				level,
				{first_layer, view_layer_count},
				actual_aspects,
				vk::ImageUsageFlagBits::eDepthStencilAttachment
			);
			// clang-format on

			vk::ResolveModeFlagBits depth_resolve_mode = vk::ResolveModeFlagBits::eNone;
			vk::ResolveModeFlagBits stencil_resolve_mode = vk::ResolveModeFlagBits::eNone;
			vk::ImageView resolve_view = nullptr;
			if (params.depth_stencil_attachment->resolve)
			{
				if (enable_depth)
				{
					depth_resolve_mode =
						params.depth_stencil_attachment->resolve->depth_mode ?
							*params.depth_stencil_attachment->resolve->depth_mode :
							vk::ResolveModeFlagBits::eSampleZero;
				}
				if (enable_stencil)
				{
					stencil_resolve_mode =
						params.depth_stencil_attachment->resolve->stencil_mode ?
							*params.depth_stencil_attachment->resolve->stencil_mode :
							vk::ResolveModeFlagBits::eSampleZero;
				}

				uint32_t resolve_level = params.depth_stencil_attachment->resolve->level ? *params.depth_stencil_attachment->resolve->level : 0;

				handle_implicit_extent(*params.depth_stencil_attachment->resolve->image, resolve_level);

				uint32_t resolve_first_layer = params.depth_stencil_attachment->resolve->first_layer ? *params.depth_stencil_attachment->resolve->first_layer : 0;

				// clang-format off
				resolve_view = (*params.depth_stencil_attachment->resolve->image)->getAttachmentView(
					(*params.depth_stencil_attachment->resolve->image)->getDesc().format,
					resolve_level,
					{resolve_first_layer, view_layer_count},
					actual_aspects,
					vk::ImageUsageFlagBits::eDepthStencilAttachment
				);
				// clang-format on
			}

			if (enable_depth)
			{
				vk::ClearDepthStencilValue clear_value;
				if (*params.depth_stencil_attachment->load_op == vk::AttachmentLoadOp::eClear)
				{
					clear_value.depth = params.depth_stencil_attachment->clear_depth_value.value();
				}

				auto& vk_attachment = vk_depth_attachment.emplace();
				vk_attachment.imageView = view;
				vk_attachment.imageLayout = vk::ImageLayout::eGeneral;
				vk_attachment.resolveMode = depth_resolve_mode;
				vk_attachment.resolveImageView = resolve_view;
				vk_attachment.resolveImageLayout = vk::ImageLayout::eGeneral;
				vk_attachment.loadOp = *params.depth_stencil_attachment->load_op;
				vk_attachment.storeOp = *params.depth_stencil_attachment->store_op;
				vk_attachment.clearValue.depthStencil = clear_value;
			}
			if (enable_stencil)
			{
				vk::ClearDepthStencilValue clear_value;
				if (*params.depth_stencil_attachment->load_op == vk::AttachmentLoadOp::eClear)
				{
					clear_value.stencil = params.depth_stencil_attachment->clear_stencil_value.value();
				}

				auto& vk_attachment = vk_stencil_attachment.emplace();
				vk_attachment.imageView = view;
				vk_attachment.imageLayout = vk::ImageLayout::eGeneral;
				vk_attachment.resolveMode = stencil_resolve_mode;
				vk_attachment.resolveImageView = resolve_view;
				vk_attachment.resolveImageLayout = vk::ImageLayout::eGeneral;
				vk_attachment.loadOp = *params.depth_stencil_attachment->load_op;
				vk_attachment.storeOp = *params.depth_stencil_attachment->store_op;
				vk_attachment.clearValue.depthStencil = clear_value;
			}
		}
	}

	vk::Rect2D render_area;
	if (params.render_area)
	{
		render_area.offset.x = static_cast<int32_t>(params.render_area->offset.x);
		render_area.offset.y = static_cast<int32_t>(params.render_area->offset.y);
		render_area.extent.width = params.render_area->size.x;
		render_area.extent.height = params.render_area->size.y;
	}
	else
	{
		if (!implicit_extent)
		{
			throw std::logic_error("If there is no attachment, render_area must be set.");
		}
		if (different_extents)
		{
			throw std::logic_error("If attachments have different extents, render_area must be set");
		}

		render_area.offset.x = 0;
		render_area.offset.y = 0;
		render_area.extent.width = implicit_extent->x;
		render_area.extent.height = implicit_extent->y;
	}

	if (render_area.extent.width == 0 || render_area.extent.height == 0)
	{
		return;
	}

	struct CmdCallback final : CmdCallbackBase
	{
		vk::CommandBuffer pass_cmd_buf;

		void operator()([[maybe_unused]] const QueuePtr& queue, vk::CommandBuffer cmd_buf, const vk::detail::DispatchLoaderDynamic& dispatcher) override
		{
			REGIONED_COMMAND_EXECUTE_BEGIN(graphicsPass)

			{
				VULKAN_CALL(vkCmdExecuteCommands);
				cmd_buf.executeCommands(
					pass_cmd_buf,
					dispatcher
				);
			}

			REGIONED_COMMAND_EXECUTE_END
		}
	};

	auto& cmd = addCmd<CmdCallback>(true);

	// With auto sync, we need to record which resources are used during the
	// pass callback and then inject the barrier before the pass is executed.
	// Using a secondary cmd buf is the only way to do that without switching
	// pass commands to slower deferred recording.
	cmd.pass_cmd_buf = m_slot->createCommandBuffer(m_queue, vk::CommandBufferLevel::eSecondary);

	{
		vk::StructureChain<
			vk::CommandBufferInheritanceInfo,
			vk::CommandBufferInheritanceDescriptorHeapInfoEXT>
			inherit_chain;

		auto& inherit_info = inherit_chain.get<vk::CommandBufferInheritanceInfo>();
		// inherit_info.renderPass;
		// inherit_info.subpass;
		// inherit_info.framebuffer;
		inherit_info.occlusionQueryEnable = vk::False;
		inherit_info.queryFlags = {};
		inherit_info.pipelineStatistics = {};

		auto& inherit_desc_heap_info = inherit_chain.get<vk::CommandBufferInheritanceDescriptorHeapInfoEXT>();
		inherit_desc_heap_info.pSamplerHeapBindInfo = &m_slot->getDeviceSession()->getSamplerBindHeapInfo();
		inherit_desc_heap_info.pResourceHeapBindInfo = &m_slot->getDeviceSession()->getResourceBindHeapInfo();

		vk::CommandBufferBeginInfo info;
		info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
		info.pInheritanceInfo = &inherit_chain.get();

		VULKAN_CALL(vkBeginCommandBuffer);
		cmd.pass_cmd_buf.begin(
			info,
			*m_dispatcher
		);
	}

	{
		vk::RenderingInfo info;
		info.flags = {};
		info.renderArea = render_area;
		info.layerCount = layer_count;
		info.viewMask = view_mask;
		info.colorAttachmentCount = static_cast<uint32_t>(vk_color_attachments.size());
		info.pColorAttachments = vk_color_attachments.data();
		info.pDepthAttachment = vk_depth_attachment ? &*vk_depth_attachment : nullptr;
		info.pStencilAttachment = vk_stencil_attachment ? &*vk_stencil_attachment : nullptr;

		VULKAN_CALL(vkCmdBeginRendering);
		cmd.pass_cmd_buf.beginRendering(
			info,
			*m_dispatcher
		);
	}

	GraphicsPassContext ctx{*this, m_slot->getDeviceSession(), cmd.pass_cmd_buf};

	ctx.setViewport({
		.x = static_cast<float>(render_area.offset.x),
		.y = static_cast<float>(render_area.offset.y),
		.width = static_cast<float>(render_area.extent.width),
		.height = static_cast<float>(render_area.extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	});

	ctx.setScissor(render_area);

	std::exception_ptr exception_ptr;
	try
	{
		(*params.callback)(ctx);
	}
	catch (...)
	{
		exception_ptr = std::current_exception();
	}

	{
		vk::RenderingEndInfoKHR info;

		VULKAN_CALL(vkCmdEndRendering);
		cmd.pass_cmd_buf.endRendering2KHR(
			info,
			*m_dispatcher
		);
	}

	{
		VULKAN_CALL(vkEndCommandBuffer);
		cmd.pass_cmd_buf.end(
			*m_dispatcher
		);
	}

	auto load_store_ops_to_accesses = [&](vk::AttachmentLoadOp load_op, vk::AttachmentStoreOp store_op, bool is_color) {
		vk::AccessFlags2 accesses;

		switch (load_op)
		{
		case vk::AttachmentLoadOp::eLoad:
			accesses |= is_color ? vk::AccessFlagBits2::eColorAttachmentRead : vk::AccessFlagBits2::eDepthStencilAttachmentRead;
			break;
		case vk::AttachmentLoadOp::eClear:
		case vk::AttachmentLoadOp::eDontCare:
			accesses |= is_color ? vk::AccessFlagBits2::eColorAttachmentWrite : vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
			break;
		case vk::AttachmentLoadOp::eNone:
			break;
		}

		switch (store_op)
		{
		case vk::AttachmentStoreOp::eStore:
		case vk::AttachmentStoreOp::eDontCare:
			accesses |= is_color ? vk::AccessFlagBits2::eColorAttachmentWrite : vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
			break;
		case vk::AttachmentStoreOp::eNone:
			break;
		}

		return accesses;
	};

	for (const auto& attachment : color_attachments)
	{
		addCmdResource(
			*attachment.image,
			{
				vk::PipelineStageFlagBits2::eColorAttachmentOutput,
				load_store_ops_to_accesses(*attachment.load_op, *attachment.store_op, true),
			}
		);

		if (attachment.resolve)
		{
			addCmdResource(
				*attachment.resolve->image,
				{
					vk::PipelineStageFlagBits2::eColorAttachmentOutput,
					vk::AccessFlagBits2::eColorAttachmentWrite,
				}
			);
		}
	}

	if (vk_depth_attachment || vk_stencil_attachment)
	{
		addCmdResource(
			*params.depth_stencil_attachment->image,
			{
				vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
				load_store_ops_to_accesses(*params.depth_stencil_attachment->load_op, *params.depth_stencil_attachment->store_op, false),
			}
		);

		if (params.depth_stencil_attachment->resolve)
		{
			addCmdResource(
				*params.depth_stencil_attachment->resolve->image,
				{
					vk::PipelineStageFlagBits2::eColorAttachmentOutput,
					vk::AccessFlagBits2::eColorAttachmentWrite,
				}
			);
		}
	}

	if (exception_ptr)
	{
		std::rethrow_exception(exception_ptr);
	}
}

void cgpu::CommandRecorder::computePass(ComputePassParams&& params)
{
	COMMAND_PARSE

	struct CmdCallback final : CmdCallbackBase
	{
		detail::BumpList<ComputePassContext::DispatchCmd> dispatch_cmds;

		explicit CmdCallback(detail::BumpMemoryResource& bump_memory):
			dispatch_cmds{detail::BumpAllocator{bump_memory}}
		{}

		void operator()([[maybe_unused]] const QueuePtr& queue, vk::CommandBuffer cmd_buf, const vk::detail::DispatchLoaderDynamic& dispatcher) override
		{
			REGIONED_COMMAND_EXECUTE_BEGIN(computePass)

			{
				std::optional<ComputeShaderStatePtr> current_compute_shader_state;
				for (const auto& dispatch_cmd : dispatch_cmds)
				{
					if (dispatch_cmd.compute_shader_state != current_compute_shader_state)
					{
						current_compute_shader_state = dispatch_cmd.compute_shader_state;

						{
							VULKAN_CALL(vkCmdBindPipeline);
							cmd_buf.bindPipeline(
								vk::PipelineBindPoint::eCompute,
								dispatch_cmd.compute_shader_state->getHandle(),
								dispatcher
							);
						}
					}

					{
						vk::PushDataInfoEXT info;
						info.offset = 0;
						info.data.address = &dispatch_cmd.params_gpu_ptr;
						info.data.size = sizeof(vk::DeviceAddress);

						{
							VULKAN_CALL(vkCmdPushDataEXT);
							cmd_buf.pushDataEXT(
								info,
								dispatcher
							);
						}
					}

					{
						VULKAN_CALL(vkCmdDispatch);
						cmd_buf.dispatch(
							dispatch_cmd.group_count.x,
							dispatch_cmd.group_count.y,
							dispatch_cmd.group_count.z,
							dispatcher
						);
					}
				}
			}

			REGIONED_COMMAND_EXECUTE_END
		}
	};

	auto& cmd = addCmd<CmdCallback>(true, *m_bump_memory);

	ComputePassContext ctx{*this, cmd.dispatch_cmds};
	std::exception_ptr exception_ptr;
	try
	{
		(*params.callback)(ctx);
	}
	catch (...)
	{
		exception_ptr = std::current_exception();
	}

	if (exception_ptr)
	{
		std::rethrow_exception(exception_ptr);
	}
}

void cgpu::CommandRecorder::buildBLAS(BLASParams&& params)
{
	COMMAND_PARSE

	struct CmdCallback final : CmdCallbackBase
	{
		BLAS::VkStructs vk_structs;
		vk::AccelerationStructureBuildRangeInfoKHR range_info;

		void operator()([[maybe_unused]] const QueuePtr& queue, vk::CommandBuffer cmd_buf, const vk::detail::DispatchLoaderDynamic& dispatcher) override
		{
			REGIONED_COMMAND_EXECUTE_BEGIN(buildBLAS)

			{
				VULKAN_CALL(vkCmdBuildAccelerationStructuresKHR);
				cmd_buf.buildAccelerationStructuresKHR(
					vk_structs.build_geometry_info,
					&range_info,
					dispatcher
				);
			}

			REGIONED_COMMAND_EXECUTE_END
		}
	};

	auto& cmd = addCmd<CmdCallback>(true);

	auto vertex_range = std::get<0>(resolveRange(*params.vertex_buffer->buffer, params.vertex_buffer->range.value_or(BufferRange{})));

	addCmdResource(
		*params.vertex_buffer->buffer,
		{
			vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
			vk::AccessFlagBits2::eShaderRead,
		}
	);

	cgpu::Range<vk::DeviceSize> index_range;
	if (params.index_buffer)
	{
		index_range = std::get<0>(resolveRange(*params.index_buffer->buffer, params.index_buffer->range.value_or(BufferRange{})));

		addCmdResource(
			*params.index_buffer->buffer,
			{
				vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
				vk::AccessFlagBits2::eShaderRead,
			}
		);
	}

	cgpu::Range<vk::DeviceSize> scratch_range;
	if (params.scratch_buffer)
	{
		scratch_range = std::get<0>(resolveRange(*params.scratch_buffer->buffer, params.scratch_buffer->range.value_or(BufferRange{})));

		addCmdResource(
			*params.scratch_buffer->buffer,
			{
				vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
				vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
			}
		);
	}

	addCmdResource(
		(*params.blas)->getBuffer(),
		{
			vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
			vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
		}
	);

	BLAS::fillVkStructs((*params.blas)->getDesc().as_info, cmd.vk_structs);

	cmd.vk_structs.geometry_info.geometry.triangles.vertexData.deviceAddress = (*params.vertex_buffer->buffer)->getDevicePtr(vertex_range.offset);
	cmd.vk_structs.geometry_info.geometry.triangles.indexData.deviceAddress = params.index_buffer ? (*params.index_buffer->buffer)->getDevicePtr(index_range.offset) : 0;

	cmd.vk_structs.build_geometry_info.dstAccelerationStructure = (*params.blas)->getHandle();
	cmd.vk_structs.build_geometry_info.scratchData.deviceAddress = params.scratch_buffer ? (*params.scratch_buffer->buffer)->getDevicePtr(scratch_range.offset) : 0;

	cmd.range_info.primitiveCount = cmd.vk_structs.primitive_count;
	cmd.range_info.primitiveOffset = 0;
	cmd.range_info.firstVertex = 0;
	cmd.range_info.transformOffset = 0;

	addReferencedObject(*params.blas);
}

void cgpu::CommandRecorder::buildTLAS(TLASParams&& params)
{
	COMMAND_PARSE

	struct CmdCallback final : CmdCallbackBase
	{
		TLAS::VkStructs vk_structs;
		vk::AccelerationStructureBuildRangeInfoKHR range_info;

		void operator()([[maybe_unused]] const QueuePtr& queue, vk::CommandBuffer cmd_buf, const vk::detail::DispatchLoaderDynamic& dispatcher) override
		{
			REGIONED_COMMAND_EXECUTE_BEGIN(buildTLAS)

			{
				VULKAN_CALL(vkCmdBuildAccelerationStructuresKHR);
				cmd_buf.buildAccelerationStructuresKHR(
					vk_structs.build_geometry_info,
					&range_info,
					dispatcher
				);
			}

			REGIONED_COMMAND_EXECUTE_END
		}
	};

	auto& cmd = addCmd<CmdCallback>(true);

	cgpu::Range<vk::DeviceSize> instance_range;
	if (params.instance_info)
	{
		instance_range = std::get<0>(resolveRange(*params.instance_info->buffer->buffer, params.instance_info->buffer->range.value_or(BufferRange{})));

		assert(instance_range.size == params.instance_info->data->size() * sizeof(vk::AccelerationStructureInstanceKHR));
		assert(((*params.instance_info->buffer->buffer)->getDevicePtr() + instance_range.offset) % 16 == 0);

		auto* instance_ptr = (*params.instance_info->buffer->buffer)->getHostPtr<vk::AccelerationStructureInstanceKHR>(instance_range.offset);
		for (const auto& instance : *params.instance_info->data)
		{
			std::memcpy(instance_ptr->transform.matrix.data()->data(), glm::value_ptr(glm::transpose(*instance.local_to_world)), sizeof(glm::mat3x4));
			instance_ptr->instanceCustomIndex = instance.custom_index.value_or(0);
			instance_ptr->mask = instance.mask.value_or(0xFF);
			instance_ptr->instanceShaderBindingTableRecordOffset = instance.sbt_record_offset.value_or(0);
			instance_ptr->flags = static_cast<VkGeometryInstanceFlagsKHR>(instance.flags.value_or(vk::GeometryInstanceFlagsKHR{}));
			instance_ptr->accelerationStructureReference = (*instance.blas)->getDevicePtr();

			instance_ptr++;

			addCmdResource(
				(*instance.blas)->getBuffer(),
				{
					vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
					vk::AccessFlagBits2::eShaderRead,
				}
			);
		}

		addCmdResource(
			*params.instance_info->buffer->buffer,
			{
				vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
				vk::AccessFlagBits2::eShaderRead,
			}
		);
	}

	cgpu::Range<vk::DeviceSize> scratch_range;
	if (params.scratch_buffer)
	{
		scratch_range = std::get<0>(resolveRange(*params.scratch_buffer->buffer, params.scratch_buffer->range.value_or(BufferRange{})));

		addCmdResource(
			*params.scratch_buffer->buffer,
			{
				vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
				vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
			}
		);
	}

	addCmdResource(
		(*params.tlas)->getBuffer(),
		{
			vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
			vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
		}
	);

	TLAS::fillVkStructs((*params.tlas)->getDesc().as_info, cmd.vk_structs);

	cmd.vk_structs.geometry_info.geometry.instances.data.deviceAddress = params.instance_info ? (*params.instance_info->buffer->buffer)->getDevicePtr(instance_range.offset) : 0;

	cmd.vk_structs.build_geometry_info.dstAccelerationStructure = (*params.tlas)->getHandle();
	cmd.vk_structs.build_geometry_info.scratchData.deviceAddress = params.scratch_buffer ? (*params.scratch_buffer->buffer)->getDevicePtr(scratch_range.offset) : 0;

	cmd.range_info.primitiveCount = cmd.vk_structs.primitive_count;
	cmd.range_info.primitiveOffset = 0;
	cmd.range_info.firstVertex = 0;
	cmd.range_info.transformOffset = 0;

	if (params.instance_info)
	{
		for (const auto& instance : *params.instance_info->data)
		{
			addReferencedObject(*instance.blas);
		}
	}

	addReferencedObject(*params.tlas);
}

void cgpu::CommandRecorder::debugBarrier(DebugBarrierParams&& params)
{
	COMMAND_PARSE

	struct CmdCallback final : CmdCallbackBase
	{
		vk::MemoryBarrier2 barrier;
		vk::DependencyInfo info;

		void operator()([[maybe_unused]] const QueuePtr& queue, vk::CommandBuffer cmd_buf, const vk::detail::DispatchLoaderDynamic& dispatcher) override
		{
			REGIONED_COMMAND_EXECUTE_BEGIN(debugBarrier)

			{
				VULKAN_CALL(vkCmdPipelineBarrier2);
				cmd_buf.pipelineBarrier2(
					info,
					dispatcher
				);
			}

			REGIONED_COMMAND_EXECUTE_END
		}
	};

	auto& cmd = addCmd<CmdCallback>(false);

	cmd.barrier.srcStageMask = params.src_stages ? *params.src_stages : vk::PipelineStageFlagBits2::eAllCommands;
	cmd.barrier.srcAccessMask = params.src_accesses ? *params.src_accesses : vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	cmd.barrier.dstStageMask = params.dst_stages ? *params.dst_stages : vk::PipelineStageFlagBits2::eAllCommands;
	cmd.barrier.dstAccessMask = params.dst_accesses ? *params.dst_accesses : vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;

	cmd.info.dependencyFlags = {};
	cmd.info.memoryBarrierCount = 1;
	cmd.info.pMemoryBarriers = &cmd.barrier;
	cmd.info.bufferMemoryBarrierCount = 0;
	// cmd.info.pBufferMemoryBarriers;
	cmd.info.imageMemoryBarrierCount = 0;
	// cmd.info.pImageMemoryBarriers;
}

void cgpu::CommandRecorder::resolve(ResolveParams&& params)
{
	COMMAND_PARSE

	vk::ImageAspectFlags aspects_in_ranges;
	std::span<const ResolveParams::Range> ranges = params.ranges ? std::span{std::as_const(*params.ranges)} : RESOLVE_DEFAULT_RANGE;
	detail::BumpVector<vk::ImageResolve2> vk_regions{detail::BumpAllocator{*m_bump_memory}};
	vk_regions.reserve(ranges.size());
	for (const auto& range : ranges)
	{
		auto [src_vk_range, src_pixel_range, src_byte_size] = resolveRange(*params.src_image, range.src.value_or(ImageLevelLayersAspectsPixelsRange{}));
		auto [dst_vk_range, dst_pixel_range, dst_byte_size] = resolveRange(*params.dst_image, range.dst.value_or(ImageLevelLayersAspectsPixelsRange{}));

		if (src_vk_range.layerCount != dst_vk_range.layerCount)
		{
			throw std::logic_error("Image ranges must have the same number of layers.");
		}

		if (src_vk_range.aspectMask != dst_vk_range.aspectMask)
		{
			throw std::logic_error("Image ranges must have the same aspects.");
		}

		if (dst_byte_size == 0)
		{
			continue;
		}

		vk::ImageResolve2& vk_region = vk_regions.emplace_back();
		vk_region.srcSubresource = src_vk_range;
		vk_region.srcOffset.x = static_cast<int>(src_pixel_range.offset.x);
		vk_region.srcOffset.y = static_cast<int>(src_pixel_range.offset.y);
		vk_region.srcOffset.z = static_cast<int>(src_pixel_range.offset.z);
		vk_region.dstSubresource = dst_vk_range;
		vk_region.dstOffset.x = static_cast<int>(dst_pixel_range.offset.x);
		vk_region.dstOffset.y = static_cast<int>(dst_pixel_range.offset.y);
		vk_region.dstOffset.z = static_cast<int>(dst_pixel_range.offset.z);
		vk_region.extent.width = src_pixel_range.size.x;
		vk_region.extent.height = src_pixel_range.size.y;
		vk_region.extent.depth = src_pixel_range.size.z;

		aspects_in_ranges |= src_vk_range.aspectMask;
	}

	if (vk_regions.empty())
	{
		return;
	}

	struct CmdCallback final : CmdCallbackBase
	{
		detail::BumpVector<vk::ImageResolve2> regions;
		vk::StructureChain<
			vk::ResolveImageInfo2,
			vk::ResolveImageModeInfoKHR>
			chain;

		explicit CmdCallback(detail::BumpVector<vk::ImageResolve2>&& regions):
			regions{std::move(regions)}
		{}

		void operator()([[maybe_unused]] const QueuePtr& queue, vk::CommandBuffer cmd_buf, const vk::detail::DispatchLoaderDynamic& dispatcher) override
		{
			REGIONED_COMMAND_EXECUTE_BEGIN(resolve)

			{
				VULKAN_CALL(vkCmdResolveImage2);
				cmd_buf.resolveImage2(
					chain.get(),
					dispatcher
				);
			}

			REGIONED_COMMAND_EXECUTE_END
		}
	};

	auto& cmd = addCmd<CmdCallback>(true, std::move(vk_regions));

	auto& resolve_info = cmd.chain.get<vk::ResolveImageInfo2>();
	resolve_info.srcImage = (*params.src_image)->getHandle();
	resolve_info.srcImageLayout = vk::ImageLayout::eGeneral;
	resolve_info.dstImage = (*params.dst_image)->getHandle();
	resolve_info.dstImageLayout = vk::ImageLayout::eGeneral;
	resolve_info.regionCount = static_cast<uint32_t>(cmd.regions.size());
	resolve_info.pRegions = cmd.regions.data();

	auto& resolve_mode_info = cmd.chain.get<vk::ResolveImageModeInfoKHR>();
	resolve_mode_info.flags = {};
	resolve_mode_info.resolveMode = vk::ResolveModeFlagBits::eNone;
	resolve_mode_info.stencilResolveMode = vk::ResolveModeFlagBits::eNone;
	if (aspects_in_ranges & vk::ImageAspectFlagBits::eColor)
	{
		resolve_mode_info.resolveMode = params.color_mode ? *params.color_mode : vk::ResolveModeFlagBits::eAverage;
	}
	if (aspects_in_ranges & vk::ImageAspectFlagBits::eDepth)
	{
		resolve_mode_info.resolveMode = params.depth_mode ? *params.depth_mode : vk::ResolveModeFlagBits::eSampleZero;
	}
	if (aspects_in_ranges & vk::ImageAspectFlagBits::eStencil)
	{
		resolve_mode_info.stencilResolveMode = params.stencil_mode ? *params.stencil_mode : vk::ResolveModeFlagBits::eSampleZero;
	}

	addCmdResource(
		*params.src_image,
		{
			vk::PipelineStageFlagBits2::eResolve,
			vk::AccessFlagBits2::eTransferRead,
		}
	);
	addCmdResource(
		*params.dst_image,
		{
			vk::PipelineStageFlagBits2::eResolve,
			vk::AccessFlagBits2::eTransferWrite,
		}
	);
}

// NOLINTEND(*-rvalue-reference-param-not-moved)

cgpu::CommandRecorder::CommandRecorder(
	std::shared_ptr<CommandContext::Slot>&& slot,
	detail::BumpMemoryResource& bump_memory,
	const QueuePtr& queue
):
	m_slot{std::move(slot)},
	m_dispatcher{&m_slot->getDeviceSession()->getDispatcher()},
	m_bump_memory{&bump_memory},
	m_queue{queue},
	m_containers{bump_memory}
{
	ZoneScoped;

	addReferencedObject(m_slot);
}

template<class T>
requires(std::derived_from<T, cgpu::Resource>)
void cgpu::CommandRecorder::addCmdResource(const std::shared_ptr<T>& resource, AccessPoints access_point)
{
	assert(!m_containers->cmd_list.empty());

	auto& curr_cmd = m_containers->cmd_list.back();
	auto [cmd_it, cmd_inserted] = curr_cmd.referenced_resources.try_emplace(resource.get(), access_point);
	if (cmd_inserted)
	{
		bool has_write_access = hasWriteAccesses(access_point.accesses);

		auto [global_it, global_inserted] = m_containers->referenced_objects.try_emplace(resource, has_write_access);
		if (global_inserted)
		{
			m_num_resources++;
		}
		else
		{
			bool& was_previously_written = global_it->second;
			if (was_previously_written || has_write_access)
			{
				m_num_resource_barriers++;
			}

			was_previously_written |= has_write_access;
		}
	}
	else
	{
		cmd_it->second |= access_point;
	}
}

template void cgpu::CommandRecorder::addCmdResource<cgpu::Image>(const std::shared_ptr<cgpu::Image>& resource, AccessPoints access_point);
template void cgpu::CommandRecorder::addCmdResource<cgpu::Buffer>(const std::shared_ptr<cgpu::Buffer>& resource, AccessPoints access_point);

template<class TCallback, class... TArgs>
requires(std::derived_from<TCallback, cgpu::CommandRecorder::CmdCallbackBase>)
TCallback& cgpu::CommandRecorder::addCmd(bool is_stageful, TArgs&&... args)
{
	Cmd& cmd = m_containers->cmd_list.emplace_back(*m_bump_memory);
	cmd.callback = detail::makeBumpUnique<TCallback>(*m_bump_memory, std::forward<TArgs>(args)...);
	cmd.is_stageful = is_stageful;

	return static_cast<TCallback&>(*cmd.callback);
}

vk::DeviceAddress cgpu::CommandRecorder::writeParameters(
	const void* data,
	size_t size,
	size_t alignment
)
{
	auto param_mem = m_slot->allocParameterMemory(size, alignment);
	std::memcpy(param_mem.cpu_ptr, data, size);

	return param_mem.gpu_ptr;
}

void cgpu::CommandRecorder::beginDebugRegion(std::string_view name, glm::vec4 color)
{
	COMMAND_PARSE

	struct CmdCallback final : CmdCallbackBase
	{
		std::string name;
		vk::DebugUtilsLabelEXT info;

		explicit CmdCallback(std::string_view name):
			name{name}
		{}

		void operator()([[maybe_unused]] const QueuePtr& queue, vk::CommandBuffer cmd_buf, const vk::detail::DispatchLoaderDynamic& dispatcher) override
		{
			VULKAN_CALL(vkCmdBeginDebugUtilsLabelEXT);
			cmd_buf.beginDebugUtilsLabelEXT(
				info,
				dispatcher
			);
		}
	};

	auto& cmd = addCmd<CmdCallback>(false, name);
	cmd.info.pLabelName = cmd.name.c_str();
	std::memcpy(cmd.info.color.data(), glm::value_ptr(color), sizeof(glm::vec4));
}

void cgpu::CommandRecorder::endDebugRegion()
{
	COMMAND_PARSE

	struct CmdCallback final : CmdCallbackBase
	{
		void operator()([[maybe_unused]] const QueuePtr& queue, vk::CommandBuffer cmd_buf, const vk::detail::DispatchLoaderDynamic& dispatcher) override
		{
			VULKAN_CALL(vkCmdEndDebugUtilsLabelEXT);
			cmd_buf.endDebugUtilsLabelEXT(
				dispatcher
			);
		}
	};

	addCmd<CmdCallback>(false);
}

cgpu::ScopedDebugRegion::ScopedDebugRegion(CommandRecorder& rec, std::string_view name, glm::vec4 color):
	m_rec{&rec}
{
	m_rec->beginDebugRegion(name, color);
}

cgpu::ScopedDebugRegion::~ScopedDebugRegion()
{
	m_rec->endDebugRegion();
}
