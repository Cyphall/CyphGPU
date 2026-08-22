#include "CommandContext.hpp"

#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/CommandRecorder.hpp>
#include <CyphGPU/Device.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/Queue.hpp>

#include <ranges>
#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>

namespace
{
constexpr vk::DeviceSize PARAMETER_BUFFER_SIZE = 65536;
constexpr vk::DeviceSize PARAMETER_BUFFER_ALIGNMENT = 256;
}

cgpu::CommandContext::CommandContext(const DeviceSessionPtr& device_session):
	m_device_session{device_session}
{
	ZoneScoped;

	beginSlot();
}

cgpu::CommandRecorder cgpu::CommandContext::createRecorder(const QueuePtr& queue)
{
	ZoneScoped;

	return m_current_slot->createRecorder(queue);
}

void cgpu::CommandContext::finish()
{
	ZoneScoped;

	endSlot();
	recycleFinishedSlots();
	beginSlot();
}

cgpu::CommandContext::Slot::Slot(const DeviceSessionPtr& device_session):
	m_device_session{device_session},
	m_min_param_buf_alloc_alignment{m_device_session->getDevice()->getProperties<vk::PhysicalDeviceProperties2>().properties.limits.minUniformBufferOffsetAlignment}
{
	ZoneScoped;
}

cgpu::CommandContext::Slot::~Slot()
{
	ZoneScoped;

	for (const auto& pool_data : m_pools | std::views::values)
	{
		{
			ZoneScopedN("vkDestroyCommandPool");
			m_device_session->getHandle().destroyCommandPool(pool_data.pool, nullptr, m_device_session->getDispatcher());
		}
	}
}

cgpu::CommandRecorder cgpu::CommandContext::Slot::createRecorder(const QueuePtr& queue)
{
	ZoneScoped;

	m_num_cmd_rec++;
	if (m_num_cmd_rec == 500 && !m_high_cmd_recs_warning_emitted)
	{
		spdlog::warn("500 command recorders were created in this slot. Did you forget to call finish() on the command context?");
		m_high_cmd_recs_warning_emitted = true;
	}

	return CommandRecorder{shared_from_this(), m_bump_memory, queue};
}

const cgpu::DeviceSessionPtr& cgpu::CommandContext::Slot::getDeviceSession() const
{
	return m_device_session;
}

cgpu::CommandContext::Slot::ParameterMemory cgpu::CommandContext::Slot::allocParameterMemory(vk::DeviceSize size, vk::DeviceSize alignment)
{
	if (size > PARAMETER_BUFFER_SIZE)
	{
		throw std::logic_error(std::format("Cannot allocate parameter memory with size > {}", PARAMETER_BUFFER_SIZE));
	}

	if (alignment > PARAMETER_BUFFER_ALIGNMENT)
	{
		throw std::logic_error(std::format("Cannot allocate parameter memory with alignment > {}", PARAMETER_BUFFER_ALIGNMENT));
	}

	alignment = std::max(alignment, m_min_param_buf_alloc_alignment);
	m_current_param_buf_offset = alignUp(m_current_param_buf_offset, alignment);
	if (m_current_param_buf_offset + size > PARAMETER_BUFFER_SIZE || m_used_param_bufs.empty())
	{
		BufferPtr param_buf{};
		if (m_free_param_bufs.empty())
		{
			param_buf = Buffer::create(
				m_device_session,
				{
					.name = "Parameter buffer",
					.size = PARAMETER_BUFFER_SIZE,
					.usages = vk::BufferUsageFlagBits2::eUniformBuffer,
					.memory_type = MemoryType::eCPUVisibleGPU,
					.min_alignment = PARAMETER_BUFFER_ALIGNMENT,
				}
			);
		}
		else
		{
			param_buf = std::move(m_free_param_bufs.back());
			m_free_param_bufs.pop_back();
		}

		m_used_param_bufs.emplace_back(std::move(param_buf));

		m_current_param_buf_offset = 0;
	}

	vk::DeviceSize alloc_offset = m_current_param_buf_offset;

	m_current_param_buf_offset += size;

	return {
		.cpu_ptr = m_used_param_bufs.back()->getHostPtr(alloc_offset),
		.gpu_ptr = m_used_param_bufs.back()->getDevicePtr(alloc_offset),
	};
}

const std::flat_map<vk::Semaphore, uint64_t>& cgpu::CommandContext::Slot::getFinishedSignals() const
{
	return m_finished_signals;
}

void cgpu::CommandContext::Slot::addFinishedSignal(const Queue::Signal& signal)
{
	ZoneScoped;

	auto [it, inserted] = m_finished_signals.try_emplace(signal.semaphore, signal.value);
	if (!inserted)
	{
		it->second = std::max(it->second, signal.value);
	}
}

void cgpu::CommandContext::Slot::recycle()
{
	ZoneScoped;

	for (auto& pool_data : m_pools | std::views::values)
	{
		{
			ZoneScopedN("vkResetCommandPool");
			m_device_session->getHandle().resetCommandPool(pool_data.pool, {}, m_device_session->getDispatcher());
		}

		for (size_t i = 0; i < 2; i++)
		{
			// Free cmd_bufs that were not used last run
			if (!pool_data.available_cmd_bufs[i].empty())
			{
				{
					ZoneScopedN("vkFreeCommandBuffers");
					m_device_session->getHandle().freeCommandBuffers(pool_data.pool, pool_data.available_cmd_bufs[i], m_device_session->getDispatcher());
				}
				pool_data.available_cmd_bufs[i].clear();
			}

			// Recycle last run command buffers
			std::swap(pool_data.in_use_cmd_bufs[i], pool_data.available_cmd_bufs[i]);
		}
	}

	m_num_cmd_rec = 0;

	m_free_param_bufs.clear();
	std::swap(m_free_param_bufs, m_used_param_bufs);
	m_current_param_buf_offset = 0;

	m_finished_signals.clear();

	m_bump_memory.release();
}

vk::CommandBuffer cgpu::CommandContext::Slot::createCommandBuffer(const QueuePtr& queue, vk::CommandBufferLevel level)
{
	ZoneScoped;

	auto [it, inserted] = m_pools.try_emplace(queue->getFamily());
	if (inserted)
	{
		vk::CommandPoolCreateInfo info;
		info.flags = vk::CommandPoolCreateFlagBits::eTransient;
		info.queueFamilyIndex = it->first;

		{
			ZoneScopedN("vkCreateCommandPool");
			it->second.pool = m_device_session->getHandle().createCommandPool(info, nullptr, m_device_session->getDispatcher());
		}
	}

	size_t level_index = static_cast<size_t>(level);

	if (it->second.available_cmd_bufs[level_index].empty())
	{
		vk::CommandBufferAllocateInfo info;
		info.commandPool = it->second.pool;
		info.level = level;
		info.commandBufferCount = 1;

		{
			ZoneScopedN("vkAllocateCommandBuffers");
			it->second.available_cmd_bufs[level_index].push_back(m_device_session->getHandle().allocateCommandBuffers(info, m_device_session->getDispatcher())[0]);
		}
	}

	vk::CommandBuffer cmd_buf = it->second.available_cmd_bufs[level_index].back();
	it->second.available_cmd_bufs[level_index].pop_back();

	it->second.in_use_cmd_bufs[level_index].push_back(cmd_buf);

	return cmd_buf;
}

void cgpu::CommandContext::beginSlot()
{
	ZoneScoped;

	if (m_available_slots.empty())
	{
		m_available_slots.push(
			std::make_shared<Slot>(
				m_device_session
			)
		);
	}

	m_current_slot = std::move(m_available_slots.top());
	m_available_slots.pop();
}

void cgpu::CommandContext::endSlot()
{
	ZoneScoped;

	if (m_current_slot->getFinishedSignals().empty())
	{
		m_current_slot->recycle();
		m_available_slots.push(std::move(m_current_slot));
	}
	else
	{
		m_pending_slots.emplace_back(std::move(m_current_slot));
	}
}

void cgpu::CommandContext::recycleFinishedSlots()
{
	ZoneScoped;

	std::erase_if(
		m_pending_slots,
		[&](const std::shared_ptr<Slot>& slot) {
			const auto& finished_signals = slot->getFinishedSignals();

			vk::SemaphoreWaitInfo info;
			info.flags = {};
			info.semaphoreCount = static_cast<uint32_t>(finished_signals.size());
			info.pSemaphores = finished_signals.keys().data();
			info.pValues = finished_signals.values().data();

			bool finished{};
			{
				ZoneScopedN("vkWaitSemaphores");
				finished = m_device_session->getHandle().waitSemaphores(info, 0, m_device_session->getDispatcher()) == vk::Result::eSuccess;
			}

			if (finished)
			{
				slot->recycle();
				m_available_slots.push(std::move(slot));
			}

			return finished;
		}
	);
}
