#include "CommandRecorder.hpp"

#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/CommandContextSlot.hpp>
#include <CyphGPU/ComputePassContext.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/Image.hpp>
#include <CyphGPU/Queue.hpp>

namespace
{
constexpr std::array DEFAULT_IMAGE_LEVELS_LAYERS_RANGE = {
	cgpu::CommandRecorder::ImageLevelsLayersRange{},
};

bool isRangeEmpty(const vk::ImageSubresourceRange& range)
{
	return range.levelCount == 0 || range.layerCount == 0 || range.aspectMask == vk::ImageAspectFlags{};
}

vk::ImageSubresourceRange resolveRange(
	const cgpu::ImagePtr& image,
	const cgpu::CommandRecorder::ImageLevelsLayersRange& range,
	vk::ImageAspectFlags aspects
)
{
	vk::ImageSubresourceRange vk_range;
	vk_range.aspectMask = aspects;
	if (range.levels)
	{
		vk_range.baseMipLevel = range.levels->offset;
		vk_range.levelCount = range.levels->size;
	}
	else
	{
		vk_range.baseMipLevel = 0;
		vk_range.levelCount = image->getDesc().levels;
	}
	if (range.layers)
	{
		vk_range.baseArrayLayer = range.layers->offset;
		vk_range.layerCount = range.layers->size;
	}
	else
	{
		vk_range.baseArrayLayer = 0;
		vk_range.layerCount = image->getDesc().layers;
	}

	return vk_range;
}
}

void cgpu::CommandRecorder::submit()
{
#if !defined(NDEBUG)
	m_submitted = true;
#endif

	m_cmdbuf.end(
		*m_dispatcher
	);

	Queue::SubmitSync sync = m_queue->submit(
		m_cmdbuf,
		m_wait_semaphores,
		m_wait_values,
		std::move(m_referenced_objects)
	);

	for (ImagePtr& image : m_referenced_images)
	{
		image->setSubmitSync(sync);
	}

	for (BufferPtr& buffer : m_referenced_buffers)
	{
		buffer->setSubmitSync(sync);
	}

	m_slot->addFinishedSync(sync);
}

void cgpu::CommandRecorder::clearImage(const ClearImageParams& params)
{
	assert(!m_submitted);

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

	std::span<const ImageLevelsLayersRange> ranges = params.ranges ? std::span{*params.ranges} : DEFAULT_IMAGE_LEVELS_LAYERS_RANGE;
	std::vector<vk::ImageSubresourceRange> vk_ranges;
	for (const auto& range : ranges)
	{
		vk::ImageSubresourceRange vk_range = resolveRange(*params.image, range, aspects);
		if (isRangeEmpty(vk_range))
		{
			continue;
		}

		vk_ranges.emplace_back(vk_range);
	}

	if (vk_ranges.empty())
	{
		return;
	}

	if (params.color_value)
	{
		vk::ClearColorValue clear_value = std::visit(
			[](auto&& value) {
				return vk::ClearColorValue{value.r, value.g, value.b, value.a};
			},
			*params.color_value
		);

		m_cmdbuf.clearColorImage(
			(*params.image)->getHandle(),
			vk::ImageLayout::eGeneral,
			clear_value,
			vk_ranges,
			*m_dispatcher
		);
	}

	if (params.depth_value || params.stencil_value)
	{
		vk::ClearDepthStencilValue clear_value = {
			params.depth_value.value_or(0.0f),
			params.stencil_value.value_or(0),
		};

		m_cmdbuf.clearDepthStencilImage(
			(*params.image)->getHandle(),
			vk::ImageLayout::eGeneral,
			clear_value,
			vk_ranges,
			*m_dispatcher
		);
	}

	addReferencedObject(*params.image);
}

void cgpu::CommandRecorder::barrier(const BarrierParams& params)
{
	assert(!m_submitted);

	vk::MemoryBarrier2 barrier;
	barrier.srcStageMask = *params.src_stages;
	barrier.srcAccessMask = *params.src_accesses;
	barrier.dstStageMask = *params.dst_stages;
	barrier.dstAccessMask = *params.dst_accesses;

	vk::DependencyInfo dep_info;
	dep_info.dependencyFlags = {};
	dep_info.memoryBarrierCount = 1;
	dep_info.pMemoryBarriers = &barrier;
	dep_info.bufferMemoryBarrierCount = 0;
	// dep_info.pBufferMemoryBarriers;
	dep_info.imageMemoryBarrierCount = 0;
	// dep_info.pImageMemoryBarriers;

	m_cmdbuf.pipelineBarrier2(
		dep_info,
		*m_dispatcher
	);
}

void cgpu::CommandRecorder::computePass(const ComputePassParams& params)
{
	assert(!m_submitted);

	ComputePassContext ctx{*this};

	(*params.callback)(ctx);
}

cgpu::CommandRecorder::CommandRecorder(
	std::shared_ptr<CommandContextSlot>&& slot,
	const cgpu::QueuePtr& queue,
	vk::CommandBuffer cmdbuf
):
	m_slot{std::move(slot)},
	m_dispatcher{&m_slot->getDeviceSession()->getDispatcher()},
	m_queue{queue},
	m_cmdbuf{cmdbuf}
{
	addReferencedObject(m_slot);

	vk::CommandBufferBeginInfo info;
	info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	// info.pInheritanceInfo;

	m_cmdbuf.begin(
		info,
		*m_dispatcher
	);

	m_cmdbuf.bindResourceHeapEXT(
		m_slot->getDeviceSession()->getResourceBindHeapInfo(),
		*m_dispatcher
	);

	m_cmdbuf.bindSamplerHeapEXT(
		m_slot->getDeviceSession()->getSamplerBindHeapInfo(),
		*m_dispatcher
	);
}

void cgpu::CommandRecorder::bindPipelineStates(
	const cgpu::ComputeShaderStatePtr& compute_shader_state
)
{
	m_cmdbuf.bindPipeline(
		vk::PipelineBindPoint::eCompute,
		compute_shader_state->getHandle(),
		*m_dispatcher
	);

	addReferencedObject(compute_shader_state);
}

void cgpu::CommandRecorder::pushParameters(
	uint32_t slot,
	const void* data,
	size_t size,
	size_t alignment
)
{
	auto param_mem = m_slot->allocParameterMemory(size, alignment);
	std::memcpy(param_mem.cpu_ptr, &data, size);

	vk::PushDataInfoEXT info;
	info.offset = slot * sizeof(vk::DeviceAddress);
	info.data.address = &param_mem.gpu_ptr;
	info.data.size = sizeof(vk::DeviceAddress);

	m_cmdbuf.pushDataEXT(
		info,
		*m_dispatcher
	);
}

void cgpu::CommandRecorder::dispatch(
	const glm::uvec3& group_count
)
{
	m_cmdbuf.dispatch(
		group_count.x,
		group_count.y,
		group_count.z,
		*m_dispatcher
	);
}

template<class T>
void cgpu::CommandRecorder::addReferencedObject(const std::shared_ptr<T>& object)
{
	m_referenced_objects.emplace_back(object);

	if constexpr (std::is_same_v<T, Image>)
	{
		m_referenced_images.emplace_back(object);
	}
	else if constexpr (std::is_same_v<T, Buffer>)
	{
		m_referenced_buffers.emplace_back(object);
	}

	if constexpr (std::is_same_v<T, Image> || std::is_same_v<T, Buffer>)
	{
		const auto& sync = object->tryGetSubmitSync();
		if (sync)
		{
			auto it = std::ranges::find(m_wait_semaphores, sync->semaphore);
			if (it == m_wait_semaphores.end())
			{
				m_wait_semaphores.emplace_back(sync->semaphore);
				m_wait_values.emplace_back(sync->value);
			}
			else
			{
				size_t index = std::distance(m_wait_semaphores.begin(), it);
				m_wait_values[index] = std::max(m_wait_values[index], sync->value);
			}
		}
	}
}
