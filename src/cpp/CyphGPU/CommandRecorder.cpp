#include "CommandRecorder.hpp"

#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/CommandContextSlot.hpp>
#include <CyphGPU/ComputePassContext.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/Image.hpp>
#include <CyphGPU/Queue.hpp>

namespace
{
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
}

void cgpu::CommandRecorder::submit()
{
#if !defined(NDEBUG)
	assert(!m_submitted);
	m_submitted = true;
#endif

	m_cmdbuf.end(
		*m_dispatcher
	);

	for (const auto& buffer : m_slot->getParameterBuffers())
	{
		addReferencedObject(buffer, ResourceAccess::eReadonly);
	}

	std::flat_map<vk::Semaphore, uint64_t> signals_to_wait;
	auto addSignalToWait = [&](vk::Semaphore semaphore, uint64_t value) {
		auto [it, inserted] = signals_to_wait.try_emplace(semaphore, value);
		if (!inserted)
		{
			it->second = std::max(it->second, value);
		}
	};

	for (auto& [resource, access] : m_referenced_resources)
	{
		resource->lock();

		switch (access)
		{
		case ResourceAccess::eReadonly:
		{
			const auto& signal = resource->tryGetReadWriteSignal();
			if (signal)
			{
				addSignalToWait(signal->semaphore, signal->value);
			}
			break;
		}
		case ResourceAccess::eReadWrite:
		{
			for (const auto& [semaphore, value] : resource->getReadSignals())
			{
				addSignalToWait(semaphore, value);
			}
			break;
		}
		default: std::unreachable();
		}
	}

	Queue::Signal signal = m_queue->submit(
		m_cmdbuf,
		signals_to_wait.keys(),
		signals_to_wait.values(),
		std::move(m_referenced_objects)
	);

	for (auto& [resource, access] : m_referenced_resources)
	{
		switch (access)
		{
		case ResourceAccess::eReadonly: resource->addReadSignal(signal); break;
		case ResourceAccess::eReadWrite: resource->setReadWriteSignal(signal); break;
		default: std::unreachable();
		}

		resource->unlock();
	}

	m_slot->addFinishedSignal(signal);
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

	std::span<const ImageLevelsLayersRange> ranges = params.ranges ? std::span{*params.ranges} : CLEAR_IMAGE_DEFAULT_RANGE;
	std::vector<vk::ImageSubresourceRange> vk_ranges;
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

	addReferencedObject(*params.image, ResourceAccess::eReadWrite);
}

void cgpu::CommandRecorder::copyImageToImage(const CopyImageToImageParams& params)
{
	assert(!m_submitted);

	std::span<const CopyImageToImageParams::Range> ranges = params.ranges ? std::span{*params.ranges} : COPY_IMAGE_TO_IMAGE_DEFAULT_RANGE;
	std::vector<vk::ImageCopy2> vk_regions;
	for (const auto& range : ranges)
	{
		auto [src_vk_range, src_pixel_range, src_byte_size] = resolveRange(*params.srcImage, range.src.value_or(ImageLevelLayersAspectsPixelsRange{}));
		auto [dst_vk_range, dst_pixel_range, dst_byte_size] = resolveRange(*params.dstImage, range.dst.value_or(ImageLevelLayersAspectsPixelsRange{}));

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

		if (src_byte_size == 0)
		{
			continue;
		}

		vk::ImageCopy2& vk_region = vk_regions.emplace_back();
		vk_region.srcSubresource = src_vk_range;
		vk_region.srcOffset.x = src_pixel_range.offset.x;
		vk_region.srcOffset.y = src_pixel_range.offset.y;
		vk_region.srcOffset.z = src_pixel_range.offset.z;
		vk_region.dstSubresource = dst_vk_range;
		vk_region.dstOffset.x = dst_pixel_range.offset.x;
		vk_region.dstOffset.y = dst_pixel_range.offset.y;
		vk_region.dstOffset.z = dst_pixel_range.offset.z;
		vk_region.extent.width = src_pixel_range.size.x;
		vk_region.extent.height = src_pixel_range.size.y;
		vk_region.extent.depth = src_pixel_range.size.z;
	}

	if (vk_regions.empty())
	{
		return;
	}

	vk::CopyImageInfo2 info;
	info.srcImage = (*params.srcImage)->getHandle();
	info.srcImageLayout = vk::ImageLayout::eGeneral;
	info.dstImage = (*params.dstImage)->getHandle();
	info.dstImageLayout = vk::ImageLayout::eGeneral;
	info.regionCount = static_cast<uint32_t>(vk_regions.size());
	info.pRegions = vk_regions.data();

	m_cmdbuf.copyImage2(
		info,
		*m_dispatcher
	);

	addReferencedObject(*params.srcImage, ResourceAccess::eReadonly);
	addReferencedObject(*params.dstImage, ResourceAccess::eReadWrite);
}

void cgpu::CommandRecorder::copyBufferToImage(const CopyBufferToImageParams& params)
{
	assert(!m_submitted);

	std::span<const CopyBufferToImageParams::Range> ranges = params.ranges ? std::span{*params.ranges} : COPY_BUFFER_TO_IMAGE_DEFAULT_RANGE;
	std::vector<vk::BufferImageCopy2> vk_regions;
	for (const auto& range : ranges)
	{
		auto [src_vk_range, src_byte_size] = resolveRange(*params.srcBuffer, range.src.value_or(BufferRange{}));
		auto [dst_vk_range, dst_pixel_range, dst_byte_size] = resolveRange(*params.dstImage, range.dst.value_or(ImageLevelLayersAspectsPixelsRange{}));

		if (src_byte_size != dst_byte_size)
		{
			throw std::logic_error("Image range and buffer range must have the same byte size.");
		}

		if (src_byte_size == 0)
		{
			continue;
		}

		vk::BufferImageCopy2& vk_region = vk_regions.emplace_back();
		vk_region.bufferOffset = src_vk_range.offset;
		vk_region.bufferRowLength = 0;
		vk_region.bufferImageHeight = 0;
		vk_region.imageSubresource = dst_vk_range;
		vk_region.imageOffset.x = dst_pixel_range.offset.x;
		vk_region.imageOffset.y = dst_pixel_range.offset.y;
		vk_region.imageOffset.z = dst_pixel_range.offset.z;
		vk_region.imageExtent.width = dst_pixel_range.size.x;
		vk_region.imageExtent.height = dst_pixel_range.size.y;
		vk_region.imageExtent.depth = dst_pixel_range.size.z;
	}

	if (vk_regions.empty())
	{
		return;
	}

	vk::CopyBufferToImageInfo2 info;
	info.srcBuffer = (*params.srcBuffer)->getHandle();
	info.dstImage = (*params.dstImage)->getHandle();
	info.dstImageLayout = vk::ImageLayout::eGeneral;
	info.regionCount = static_cast<uint32_t>(vk_regions.size());
	info.pRegions = vk_regions.data();

	m_cmdbuf.copyBufferToImage2(
		info,
		*m_dispatcher
	);

	addReferencedObject(*params.srcBuffer, ResourceAccess::eReadonly);
	addReferencedObject(*params.dstImage, ResourceAccess::eReadWrite);
}

void cgpu::CommandRecorder::copyImageToBuffer(const CopyImageToBufferParams& params)
{
	assert(!m_submitted);

	std::span<const CopyImageToBufferParams::Range> ranges = params.ranges ? std::span{*params.ranges} : COPY_IMAGE_TO_BUFFER_DEFAULT_RANGE;
	std::vector<vk::BufferImageCopy2> vk_regions;
	for (const auto& range : ranges)
	{
		auto [src_vk_range, src_pixel_range, src_byte_size] = resolveRange(*params.srcImage, range.src.value_or(ImageLevelLayersAspectsPixelsRange{}));
		auto [dst_vk_range, dst_byte_size] = resolveRange(*params.dstBuffer, range.dst.value_or(BufferRange{}));

		if (src_byte_size != dst_byte_size)
		{
			throw std::logic_error("Image range and buffer range must have the same byte size.");
		}

		if (src_byte_size == 0)
		{
			continue;
		}

		vk::BufferImageCopy2& vk_region = vk_regions.emplace_back();
		vk_region.bufferOffset = dst_vk_range.offset;
		vk_region.bufferRowLength = 0;
		vk_region.bufferImageHeight = 0;
		vk_region.imageSubresource = src_vk_range;
		vk_region.imageOffset.x = src_pixel_range.offset.x;
		vk_region.imageOffset.y = src_pixel_range.offset.y;
		vk_region.imageOffset.z = src_pixel_range.offset.z;
		vk_region.imageExtent.width = src_pixel_range.size.x;
		vk_region.imageExtent.height = src_pixel_range.size.y;
		vk_region.imageExtent.depth = src_pixel_range.size.z;
	}

	if (vk_regions.empty())
	{
		return;
	}

	vk::CopyImageToBufferInfo2 info;
	info.srcImage = (*params.srcImage)->getHandle();
	info.srcImageLayout = vk::ImageLayout::eGeneral;
	info.dstBuffer = (*params.dstBuffer)->getHandle();
	info.regionCount = static_cast<uint32_t>(vk_regions.size());
	info.pRegions = vk_regions.data();

	m_cmdbuf.copyImageToBuffer2(
		info,
		*m_dispatcher
	);

	addReferencedObject(*params.srcImage, ResourceAccess::eReadonly);
	addReferencedObject(*params.dstBuffer, ResourceAccess::eReadWrite);
}

void cgpu::CommandRecorder::copyBufferToBuffer(const CopyBufferToBufferParams& params)
{
	assert(!m_submitted);

	std::span<const CopyBufferToBufferParams::Range> ranges = params.ranges ? std::span{*params.ranges} : COPY_BUFFER_TO_BUFFER_DEFAULT_RANGE;
	std::vector<vk::BufferCopy2> vk_regions;
	for (const auto& range : ranges)
	{
		auto [src_vk_range, src_byte_size] = resolveRange(*params.srcBuffer, range.src.value_or(BufferRange{}));
		auto [dst_vk_range, dst_byte_size] = resolveRange(*params.dstBuffer, range.dst.value_or(BufferRange{}));

		if (src_byte_size != dst_byte_size)
		{
			throw std::logic_error("Buffer ranges must have the same byte size.");
		}

		if (src_byte_size == 0)
		{
			continue;
		}

		vk::BufferCopy2& vk_region = vk_regions.emplace_back();
		vk_region.srcOffset = src_vk_range.offset;
		vk_region.dstOffset = dst_vk_range.offset;
		vk_region.size = src_vk_range.size;
	}

	if (vk_regions.empty())
	{
		return;
	}

	vk::CopyBufferInfo2 info;
	info.srcBuffer = (*params.srcBuffer)->getHandle();
	info.dstBuffer = (*params.dstBuffer)->getHandle();
	info.regionCount = static_cast<uint32_t>(vk_regions.size());
	info.pRegions = vk_regions.data();

	m_cmdbuf.copyBuffer2(
		info,
		*m_dispatcher
	);

	addReferencedObject(*params.srcBuffer, ResourceAccess::eReadonly);
	addReferencedObject(*params.dstBuffer, ResourceAccess::eReadWrite);
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

template<class T>
requires(!std::derived_from<T, cgpu::Resource>)
void cgpu::CommandRecorder::addReferencedObject(const std::shared_ptr<T>& object)
{
	m_referenced_objects.emplace_back(object);
}

void cgpu::CommandRecorder::addReferencedObject(const std::shared_ptr<Resource>& resource, ResourceAccess access)
{
	m_referenced_objects.emplace_back(resource);

	auto [it, inserted] = m_referenced_resources.try_emplace(resource, access);
	if (!inserted && access == ResourceAccess::eReadWrite)
	{
		it->second = ResourceAccess::eReadWrite;
	}
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
