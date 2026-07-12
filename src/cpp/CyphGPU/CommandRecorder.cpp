#include "CommandRecorder.hpp"

#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/CommandContextSlot.hpp>
#include <CyphGPU/ComputePassContext.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/GraphicsPassContext.hpp>
#include <CyphGPU/Image.hpp>
#include <CyphGPU/Queue.hpp>

namespace
{
template<class... Ts>
struct overloaded : Ts...
{
	using Ts::operator()...;
};

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
	auto add_signal_to_wait = [&](vk::Semaphore semaphore, uint64_t value) {
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
				add_signal_to_wait(signal->semaphore, signal->value);
			}
			break;
		}
		case ResourceAccess::eReadWrite:
		{
			for (const auto& [semaphore, value] : resource->getReadSignals())
			{
				add_signal_to_wait(semaphore, value);
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

	vk::CopyImageInfo2 info;
	info.srcImage = (*params.src_image)->getHandle();
	info.srcImageLayout = vk::ImageLayout::eGeneral;
	info.dstImage = (*params.dst_image)->getHandle();
	info.dstImageLayout = vk::ImageLayout::eGeneral;
	info.regionCount = static_cast<uint32_t>(vk_regions.size());
	info.pRegions = vk_regions.data();

	m_cmdbuf.copyImage2(
		info,
		*m_dispatcher
	);

	addReferencedObject(*params.src_image, ResourceAccess::eReadonly);
	addReferencedObject(*params.dst_image, ResourceAccess::eReadWrite);
}

void cgpu::CommandRecorder::copyBufferToImage(const CopyBufferToImageParams& params)
{
	assert(!m_submitted);

	std::span<const CopyBufferToImageParams::Range> ranges = params.ranges ? std::span{*params.ranges} : COPY_BUFFER_TO_IMAGE_DEFAULT_RANGE;
	std::vector<vk::BufferImageCopy2> vk_regions;
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

		vk::BufferImageCopy2& vk_region = vk_regions.emplace_back();
		vk_region.bufferOffset = src_vk_range.offset;
		vk_region.bufferRowLength = 0;
		vk_region.bufferImageHeight = 0;
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

	vk::CopyBufferToImageInfo2 info;
	info.srcBuffer = (*params.src_buffer)->getHandle();
	info.dstImage = (*params.dst_image)->getHandle();
	info.dstImageLayout = vk::ImageLayout::eGeneral;
	info.regionCount = static_cast<uint32_t>(vk_regions.size());
	info.pRegions = vk_regions.data();

	m_cmdbuf.copyBufferToImage2(
		info,
		*m_dispatcher
	);

	addReferencedObject(*params.src_buffer, ResourceAccess::eReadonly);
	addReferencedObject(*params.dst_image, ResourceAccess::eReadWrite);
}

void cgpu::CommandRecorder::copyImageToBuffer(const CopyImageToBufferParams& params)
{
	assert(!m_submitted);

	std::span<const CopyImageToBufferParams::Range> ranges = params.ranges ? std::span{*params.ranges} : COPY_IMAGE_TO_BUFFER_DEFAULT_RANGE;
	std::vector<vk::BufferImageCopy2> vk_regions;
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

		vk::BufferImageCopy2& vk_region = vk_regions.emplace_back();
		vk_region.bufferOffset = dst_vk_range.offset;
		vk_region.bufferRowLength = 0;
		vk_region.bufferImageHeight = 0;
		vk_region.imageSubresource = src_vk_range;
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

	vk::CopyImageToBufferInfo2 info;
	info.srcImage = (*params.src_image)->getHandle();
	info.srcImageLayout = vk::ImageLayout::eGeneral;
	info.dstBuffer = (*params.dst_buffer)->getHandle();
	info.regionCount = static_cast<uint32_t>(vk_regions.size());
	info.pRegions = vk_regions.data();

	m_cmdbuf.copyImageToBuffer2(
		info,
		*m_dispatcher
	);

	addReferencedObject(*params.src_image, ResourceAccess::eReadonly);
	addReferencedObject(*params.dst_buffer, ResourceAccess::eReadWrite);
}

void cgpu::CommandRecorder::copyBufferToBuffer(const CopyBufferToBufferParams& params)
{
	assert(!m_submitted);

	std::span<const CopyBufferToBufferParams::Range> ranges = params.ranges ? std::span{*params.ranges} : COPY_BUFFER_TO_BUFFER_DEFAULT_RANGE;
	std::vector<vk::BufferCopy2> vk_regions;
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
	info.srcBuffer = (*params.src_buffer)->getHandle();
	info.dstBuffer = (*params.dst_buffer)->getHandle();
	info.regionCount = static_cast<uint32_t>(vk_regions.size());
	info.pRegions = vk_regions.data();

	m_cmdbuf.copyBuffer2(
		info,
		*m_dispatcher
	);

	addReferencedObject(*params.src_buffer, ResourceAccess::eReadonly);
	addReferencedObject(*params.dst_buffer, ResourceAccess::eReadWrite);
}

void cgpu::CommandRecorder::blit(const BlitParams& params)
{
	assert(!m_submitted);

	std::span<const BlitParams::Range> ranges = params.ranges ? std::span{*params.ranges} : BLIT_DEFAULT_RANGE;
	std::vector<vk::ImageBlit2> vk_regions;
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

	vk::BlitImageInfo2 info;
	info.srcImage = (*params.src_image)->getHandle();
	info.srcImageLayout = vk::ImageLayout::eGeneral;
	info.dstImage = (*params.dst_image)->getHandle();
	info.dstImageLayout = vk::ImageLayout::eGeneral;
	info.regionCount = static_cast<uint32_t>(vk_regions.size());
	info.pRegions = vk_regions.data();
	info.filter = params.filter.value_or(vk::Filter::eNearest);

	m_cmdbuf.blitImage2(
		info,
		*m_dispatcher
	);

	addReferencedObject(*params.src_image, ResourceAccess::eReadonly);
	addReferencedObject(*params.dst_image, ResourceAccess::eReadWrite);
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

void cgpu::CommandRecorder::graphicsPass(const GraphicsPassParams& params)
{
	assert(!m_submitted);

	uint32_t layer_count = 1;
	uint32_t view_mask = 0;
	if (params.layer_mode)
	{
		std::visit(
			overloaded{
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
	std::vector<vk::RenderingAttachmentInfo> vk_color_attachments;
	vk_color_attachments.reserve(color_attachments.size());
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

		vk::ClearColorValue clear_value;
		if (*attachment.load_op == vk::AttachmentLoadOp::eClear)
		{
			clear_value = std::visit(
				[](auto&& value) {
					return vk::ClearColorValue{value.r, value.g, value.b, value.a};
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
		vk_attachment.clearValue = clear_value;
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

			vk::ResolveModeFlagBits resolve_mode = vk::ResolveModeFlagBits::eNone;
			vk::ImageView resolve_view = nullptr;
			if (params.depth_stencil_attachment->resolve)
			{
				resolve_mode =
					params.depth_stencil_attachment->resolve->mode ?
						*params.depth_stencil_attachment->resolve->mode :
						vk::ResolveModeFlagBits::eSampleZero;

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
				vk_attachment.resolveMode = resolve_mode;
				vk_attachment.resolveImageView = resolve_view;
				vk_attachment.resolveImageLayout = vk::ImageLayout::eGeneral;
				vk_attachment.loadOp = *params.depth_stencil_attachment->load_op;
				vk_attachment.storeOp = *params.depth_stencil_attachment->store_op;
				vk_attachment.clearValue = clear_value;
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
				vk_attachment.resolveMode = resolve_mode;
				vk_attachment.resolveImageView = resolve_view;
				vk_attachment.resolveImageLayout = vk::ImageLayout::eGeneral;
				vk_attachment.loadOp = *params.depth_stencil_attachment->load_op;
				vk_attachment.storeOp = *params.depth_stencil_attachment->store_op;
				vk_attachment.clearValue = clear_value;
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

	vk::RenderingInfo rendering_info;
	rendering_info.flags = {};
	rendering_info.renderArea = render_area;
	rendering_info.layerCount = layer_count;
	rendering_info.viewMask = view_mask;
	rendering_info.colorAttachmentCount = static_cast<uint32_t>(vk_color_attachments.size());
	rendering_info.pColorAttachments = vk_color_attachments.data();
	rendering_info.pDepthAttachment = vk_depth_attachment ? &*vk_depth_attachment : nullptr;
	rendering_info.pStencilAttachment = vk_stencil_attachment ? &*vk_stencil_attachment : nullptr;

	m_cmdbuf.beginRendering(
		rendering_info,
		*m_dispatcher
	);

	vk::Viewport viewport;
	viewport.x = static_cast<float>(render_area.offset.x);
	viewport.y = static_cast<float>(render_area.offset.y);
	viewport.width = static_cast<float>(render_area.extent.width);
	viewport.height = static_cast<float>(render_area.extent.height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	m_cmdbuf.setViewport(
		0,
		viewport,
		*m_dispatcher
	);

	m_cmdbuf.setScissor(
		0,
		render_area,
		*m_dispatcher
	);

	std::exception_ptr exception_ptr;
	try
	{
		GraphicsPassContext ctx{*this};
		(*params.callback)(ctx);
	}
	catch (...)
	{
		exception_ptr = std::current_exception();
	}

	m_cmdbuf.endRendering(
		*m_dispatcher
	);

	for (const auto& attachment : color_attachments)
	{
		addReferencedObject(*attachment.image, ResourceAccess::eReadWrite);
		if (attachment.resolve)
		{
			addReferencedObject(*attachment.resolve->image, ResourceAccess::eReadWrite);
		}
	}

	if (params.depth_stencil_attachment && (vk_depth_attachment || vk_stencil_attachment))
	{
		addReferencedObject(*params.depth_stencil_attachment->image, ResourceAccess::eReadWrite);
		if (params.depth_stencil_attachment->resolve)
		{
			addReferencedObject(*params.depth_stencil_attachment->resolve->image, ResourceAccess::eReadWrite);
		}
	}

	if (exception_ptr)
	{
		std::rethrow_exception(exception_ptr);
	}
}

void cgpu::CommandRecorder::computePass(const ComputePassParams& params)
{
	assert(!m_submitted);

	{
		ComputePassContext ctx{*this};
		(*params.callback)(ctx);
	}
}

cgpu::CommandRecorder::CommandRecorder(
	std::shared_ptr<CommandContextSlot>&& slot,
	const QueuePtr& queue,
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
	const VertexInputStatePtr& vertex_input_state,
	const PreRasterizationShaderStatePtr& pre_rasterization_shader_state,
	const FragmentShaderStatePtr& fragment_shader_state,
	const FragmentOutputStatePtr& fragment_output_state
)
{
	vk::Pipeline pipeline = m_slot->getDeviceSession()->getGraphicsPipeline(
		vertex_input_state,
		pre_rasterization_shader_state,
		fragment_shader_state,
		fragment_output_state
	);

	m_cmdbuf.bindPipeline(
		vk::PipelineBindPoint::eGraphics,
		pipeline,
		*m_dispatcher
	);

	addReferencedObject(vertex_input_state);
	addReferencedObject(pre_rasterization_shader_state);
	addReferencedObject(fragment_shader_state);
	addReferencedObject(fragment_output_state);
}

void cgpu::CommandRecorder::bindPipelineStates(
	const ComputeShaderStatePtr& compute_shader_state
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
	std::memcpy(param_mem.cpu_ptr, data, size);

	vk::PushDataInfoEXT info;
	info.offset = slot * sizeof(vk::DeviceAddress);
	info.data.address = &param_mem.gpu_ptr;
	info.data.size = sizeof(vk::DeviceAddress);

	m_cmdbuf.pushDataEXT(
		info,
		*m_dispatcher
	);
}

void cgpu::CommandRecorder::draw(
	uint32_t vertex_count,
	uint32_t instance_count,
	uint32_t first_vertex,
	uint32_t first_instance
)
{
	m_cmdbuf.draw(
		vertex_count,
		instance_count,
		first_vertex,
		first_instance,
		*m_dispatcher
	);
}

void cgpu::CommandRecorder::drawIndexed(
	uint32_t index_count,
	uint32_t instance_count,
	uint32_t first_index,
	int32_t vertex_offset,
	uint32_t first_instance
)
{
	m_cmdbuf.drawIndexed(
		index_count,
		instance_count,
		first_index,
		vertex_offset,
		first_instance,
		*m_dispatcher
	);
}

void cgpu::CommandRecorder::setViewport(
	const vk::Viewport& viewport
)
{
	m_cmdbuf.setViewport(
		0,
		viewport,
		*m_dispatcher
	);
}

void cgpu::CommandRecorder::setScissor(
	const vk::Rect2D& scissor
)
{
	m_cmdbuf.setScissor(
		0,
		scissor,
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
