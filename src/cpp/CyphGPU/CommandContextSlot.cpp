#include "CommandContextSlot.hpp"

#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/Queue.hpp>
#include <CyphGPU/Utils.hpp>

#include <ranges>

namespace
{
constexpr vk::DeviceSize PARAMETER_BUFFER_SIZE = 4096;
}

cgpu::CommandContextSlot::CommandContextSlot(PrivateKey, const cgpu::DeviceSessionPtr& device_session):
	m_device_session{device_session}
{
}

cgpu::CommandContextSlot::~CommandContextSlot()
{
	for (vk::CommandPool pool : m_pools | std::views::values)
	{
		m_device_session->getHandle().destroyCommandPool(pool, nullptr, m_device_session->getDispatcher());
	}
}

cgpu::CommandRecorder cgpu::CommandContextSlot::createRecorder(const cgpu::QueuePtr& queue)
{
	auto [it, inserted] = m_pools.try_emplace(queue->getFamily());
	if (inserted)
	{
		vk::CommandPoolCreateInfo info;
		info.flags = vk::CommandPoolCreateFlagBits::eTransient;
		info.queueFamilyIndex = it->first;

		it->second = m_device_session->getHandle().createCommandPool(info, nullptr, m_device_session->getDispatcher());
	}

	vk::CommandBufferAllocateInfo info;
	info.commandPool = it->second;
	info.level = vk::CommandBufferLevel::ePrimary;
	info.commandBufferCount = 1;

	vk::CommandBuffer cmdbuf = m_device_session->getHandle().allocateCommandBuffers(info, m_device_session->getDispatcher())[0];

	return CommandRecorder{shared_from_this(), queue, cmdbuf};
}

const cgpu::DeviceSessionPtr& cgpu::CommandContextSlot::getDeviceSession() const
{
	return m_device_session;
}

cgpu::CommandContextSlot::ParameterMemory cgpu::CommandContextSlot::allocParameterMemory(vk::DeviceSize size, vk::DeviceSize alignment)
{
	if (size > PARAMETER_BUFFER_SIZE)
	{
		throw std::logic_error(std::format("Cannot request parameter size greater than {}", PARAMETER_BUFFER_SIZE));
	}

	if (alignment > PARAMETER_BUFFER_SIZE)
	{
		throw std::logic_error(std::format("Cannot request parameter alignment greater than {}", PARAMETER_BUFFER_SIZE));
	}

	vk::DeviceSize aligned_offset = alignUp(m_parameter_offset, alignment);

	vk::DeviceSize total_space = m_parameter_buffers.size() * PARAMETER_BUFFER_SIZE;
	vk::DeviceSize remaining_buffer_space = total_space - aligned_offset;
	if (remaining_buffer_space < size)
	{
		aligned_offset = total_space;

		m_parameter_buffers.emplace_back(
			Buffer::create(
				m_device_session,
				{
					.name = "Parameter buffer",
					.size = PARAMETER_BUFFER_SIZE,
					.usages = vk::BufferUsageFlagBits2::eUniformBuffer,
					.memory_type = MemoryType::eCPUVisibleGPU,
					.min_alignment = PARAMETER_BUFFER_SIZE,
				}
			)
		);
	}

	vk::DeviceSize buffer_offset = aligned_offset % PARAMETER_BUFFER_SIZE;

	m_parameter_offset = buffer_offset + size;

	return {
		.cpu_ptr = m_parameter_buffers.back()->getHostPtr(buffer_offset),
		.gpu_ptr = m_parameter_buffers.back()->getDevicePtr(buffer_offset),
	};
}

const std::flat_map<vk::Semaphore, uint64_t>& cgpu::CommandContextSlot::getFinishedSignals() const
{
	return m_finished_signals;
}

void cgpu::CommandContextSlot::addFinishedSignal(const Queue::Signal& signal)
{
	auto [it, inserted] = m_finished_signals.try_emplace(signal.semaphore, signal.value);
	if (!inserted)
	{
		it->second = std::max(it->second, signal.value);
	}
}

void cgpu::CommandContextSlot::reset()
{
	for (vk::CommandPool pool : m_pools | std::views::values)
	{
		m_device_session->getHandle().resetCommandPool(pool, {}, m_device_session->getDispatcher());
	}

	m_parameter_offset = 0;

	m_finished_signals.clear();
}

std::span<const cgpu::BufferPtr> cgpu::CommandContextSlot::getParameterBuffers()
{
	return m_parameter_buffers;
}
