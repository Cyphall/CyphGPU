#include "CommandRecorder.hpp"

#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/CommandContextSlot.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/Image.hpp>
#include <CyphGPU/Queue.hpp>

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

	std::vector<vk::ImageSubresourceRange> vk_ranges;
	if (params.ranges)
	{
		for (const auto& range : *params.ranges)
		{
			auto& vk_range = vk_ranges.emplace_back();
			vk_range.aspectMask = vk::ImageAspectFlagBits::eColor;
			vk_range.baseMipLevel = range.levels->offset;
			vk_range.levelCount = range.levels->size;
			vk_range.baseArrayLayer = range.layers->offset;
			vk_range.layerCount = range.layers->size;
		}
	}
	else
	{
		auto& vk_range = vk_ranges.emplace_back();
		vk_range.aspectMask = vk::ImageAspectFlagBits::eColor;
		vk_range.baseMipLevel = 0;
		vk_range.levelCount = (*params.image)->getDesc().levels;
		vk_range.baseArrayLayer = 0;
		vk_range.layerCount = (*params.image)->getDesc().layers;
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
