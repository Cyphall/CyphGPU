#include "CommandContextSlot.hpp"

#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/Device.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/Queue.hpp>
#include <CyphGPU/Utils.hpp>

#include <ranges>
#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>

namespace
{
constexpr vk::DeviceSize PARAMETER_BUFFER_SIZE = 65536;
constexpr vk::DeviceSize PARAMETER_BUFFER_ALIGNMENT = 256;
}

cgpu::CommandContextSlot::CommandContextSlot(PrivateKey, const DeviceSessionPtr& device_session):
	m_device_session{device_session},
	m_min_parambuf_alloc_alignment{m_device_session->getDevice()->getProperties<vk::PhysicalDeviceProperties2>().properties.limits.minUniformBufferOffsetAlignment}
{
	ZoneScoped;
}

cgpu::CommandContextSlot::~CommandContextSlot()
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

cgpu::CommandRecorder cgpu::CommandContextSlot::createRecorder(const QueuePtr& queue)
{
	ZoneScoped;

	vk::CommandBuffer cmdbuf = createCommandBuffer(queue, vk::CommandBufferLevel::ePrimary);

	m_num_cmd_rec++;
	if (m_num_cmd_rec == 500 && !m_high_cmd_recs_warning_emitted)
	{
		spdlog::warn("500 command recorders were created in this slot. Did you forget to call finish() on the command context?");
		m_high_cmd_recs_warning_emitted = true;
	}

	return CommandRecorder{shared_from_this(), m_bump_memory, queue, cmdbuf};
}

const cgpu::DeviceSessionPtr& cgpu::CommandContextSlot::getDeviceSession() const
{
	return m_device_session;
}

cgpu::CommandContextSlot::ParameterMemory cgpu::CommandContextSlot::allocParameterMemory(vk::DeviceSize size, vk::DeviceSize alignment)
{
	if (size > PARAMETER_BUFFER_SIZE)
	{
		throw std::logic_error(std::format("Cannot allocate parameter memory with size > {}", PARAMETER_BUFFER_SIZE));
	}

	if (alignment > PARAMETER_BUFFER_ALIGNMENT)
	{
		throw std::logic_error(std::format("Cannot allocate parameter memory with alignment > {}", PARAMETER_BUFFER_ALIGNMENT));
	}

	alignment = std::max(alignment, m_min_parambuf_alloc_alignment);
	m_current_parambuf_offset = alignUp(m_current_parambuf_offset, alignment);
	if (m_current_parambuf_offset + size > PARAMETER_BUFFER_SIZE || m_used_parambufs.empty())
	{
		BufferPtr parambuf{};
		if (m_free_parambufs.empty())
		{
			parambuf = Buffer::create(
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
			parambuf = std::move(m_free_parambufs.back());
			m_free_parambufs.pop_back();
		}

		m_used_parambufs.emplace_back(std::move(parambuf));

		m_current_parambuf_offset = 0;
	}

	vk::DeviceSize alloc_offset = m_current_parambuf_offset;

	m_current_parambuf_offset += size;

	return {
		.cpu_ptr = m_used_parambufs.back()->getHostPtr(alloc_offset),
		.gpu_ptr = m_used_parambufs.back()->getDevicePtr(alloc_offset),
	};
}

const std::flat_map<vk::Semaphore, uint64_t>& cgpu::CommandContextSlot::getFinishedSignals() const
{
	return m_finished_signals;
}

void cgpu::CommandContextSlot::addFinishedSignal(const Queue::Signal& signal)
{
	ZoneScoped;

	auto [it, inserted] = m_finished_signals.try_emplace(signal.semaphore, signal.value);
	if (!inserted)
	{
		it->second = std::max(it->second, signal.value);
	}
}

void cgpu::CommandContextSlot::reset()
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
			// Free cmdbufs that were not used last run
			if (!pool_data.available_cmdbufs[i].empty())
			{
				{
					ZoneScopedN("vkFreeCommandBuffers");
					m_device_session->getHandle().freeCommandBuffers(pool_data.pool, pool_data.available_cmdbufs[i], m_device_session->getDispatcher());
				}
				pool_data.available_cmdbufs[i].clear();
			}

			// Recycle last run command buffers
			std::swap(pool_data.in_use_cmdbufs[i], pool_data.available_cmdbufs[i]);
		}
	}

	m_num_cmd_rec = 0;

	m_free_parambufs.clear();
	std::swap(m_free_parambufs, m_used_parambufs);
	m_current_parambuf_offset = 0;

	m_finished_signals.clear();

	m_bump_memory.release();
}

vk::CommandBuffer cgpu::CommandContextSlot::createCommandBuffer(const QueuePtr& queue, vk::CommandBufferLevel level)
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

	if (it->second.available_cmdbufs[level_index].empty())
	{
		vk::CommandBufferAllocateInfo info;
		info.commandPool = it->second.pool;
		info.level = level;
		info.commandBufferCount = 1;

		{
			ZoneScopedN("vkAllocateCommandBuffers");
			it->second.available_cmdbufs[level_index].push_back(m_device_session->getHandle().allocateCommandBuffers(info, m_device_session->getDispatcher())[0]);
		}
	}

	vk::CommandBuffer cmdbuf = it->second.available_cmdbufs[level_index].back();
	it->second.available_cmdbufs[level_index].pop_back();

	it->second.in_use_cmdbufs[level_index].push_back(cmdbuf);

	return cmdbuf;
}
