#include "CommandContext.hpp"

#include <CyphGPU/DeviceSession.hpp>

#include <tracy/Tracy.hpp>

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

void cgpu::CommandContext::beginSlot()
{
	ZoneScoped;

	if (m_available_slots.empty())
	{
		m_available_slots.push(
			std::make_shared<CommandContextSlot>(
				CommandContextSlot::PrivateKey{},
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

	m_pending_slots.emplace_back(std::move(m_current_slot));
}

void cgpu::CommandContext::recycleFinishedSlots()
{
	ZoneScoped;

	std::erase_if(
		m_pending_slots,
		[&](const std::shared_ptr<CommandContextSlot>& slot) {
			const auto& finished_signals = slot->getFinishedSignals();

			vk::SemaphoreWaitInfo info;
			info.flags = {};
			info.semaphoreCount = static_cast<uint32_t>(finished_signals.size());
			info.pSemaphores = finished_signals.keys().data();
			info.pValues = finished_signals.values().data();

			bool finished = m_device_session->getHandle().waitSemaphores(info, 0, m_device_session->getDispatcher()) == vk::Result::eSuccess;
			if (finished)
			{
				slot->reset();
				m_available_slots.push(std::move(slot));
			}

			return finished;
		}
	);
}
