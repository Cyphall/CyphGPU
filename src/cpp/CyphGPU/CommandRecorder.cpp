#include "CommandRecorder.hpp"

#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/CommandContextSlot.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/Image.hpp>
#include <CyphGPU/Queue.hpp>

namespace
{
constexpr std::array DEFAULT_IMAGE_LEVEL_LAYER_RANGE = {
	cgpu::CommandRecorder::ImageLevelLayerRange{},
};

bool isRangeEmpty(const vk::ImageSubresourceRange& range)
{
	return range.levelCount == 0 || range.layerCount == 0 || range.aspectMask == vk::ImageAspectFlags{};
}

vk::ImageSubresourceRange resolveRange(
	const cgpu::ImagePtr& image,
	const cgpu::CommandRecorder::ImageLevelLayerRange& range,
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

void cgpu::CommandRecorder::clearColorImage(const ClearColorImageParams& params)
{
	assert(!m_submitted);

	std::span<const ImageLevelLayerRange> ranges = params.ranges ? std::span{*params.ranges} : DEFAULT_IMAGE_LEVEL_LAYER_RANGE;
	std::vector<vk::ImageSubresourceRange> vk_ranges;
	for (const auto& range : ranges)
	{
		vk::ImageSubresourceRange vk_range = resolveRange(*params.image, range, vk::ImageAspectFlagBits::eColor);
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

	vk::ClearColorValue clear_value = std::visit(
		[](auto&& value) {
			return vk::ClearColorValue{value.r, value.g, value.b, value.a};
		},
		*params.clear_value
	);

	m_cmdbuf.clearColorImage(
		(*params.image)->getHandle(),
		vk::ImageLayout::eGeneral,
		clear_value,
		vk_ranges,
		*m_dispatcher
	);

	addReferencedObject(*params.image);
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
