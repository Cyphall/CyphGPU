#include "CommandContext.hpp"

#include <CyphGPU/DeviceSession.hpp>

cgpu::CommandContext::CommandContext(const cgpu::DeviceSessionPtr& device_session):
	m_device_session{device_session}
{
	beginSlot();
}

cgpu::CommandRecorder cgpu::CommandContext::createRecorder(const cgpu::QueuePtr& queue)
{
	return m_current_slot->createRecorder(queue);
}

void cgpu::CommandContext::finish()
{
	endSlot();
	recycleFinishedSlots();
	beginSlot();
}

void cgpu::CommandContext::beginSlot()
{
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
	m_pending_slots.emplace_back(std::move(m_current_slot));
}

void cgpu::CommandContext::recycleFinishedSlots()
{
	std::erase_if(
		m_pending_slots,
		[&](const std::shared_ptr<CommandContextSlot>& slot) {
			auto [semaphores, values] = slot->getFinishedSemaphores();

			vk::SemaphoreWaitInfo info;
			info.flags = {};
			info.semaphoreCount = static_cast<uint32_t>(semaphores.size());
			info.pSemaphores = semaphores.data();
			info.pValues = values.data();

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
