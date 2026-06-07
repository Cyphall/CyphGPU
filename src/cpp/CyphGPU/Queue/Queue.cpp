#include "Queue.hpp"

#include <CyphGPU/DeviceSession/DeviceSession.hpp>

cgpu::Queue::Queue(PrivateKey, DeviceSession& device_session, vk::Queue queue):
	DependencyObject{device_session},
	m_device_session{device_session},
	m_handle{queue}
{
	createSemaphore();
}

cgpu::Queue::~Queue()
{
	m_device_session->getHandle().destroySemaphore(m_semaphore, nullptr, m_device_session->getDispatcher());
}

const vk::Queue& cgpu::Queue::getHandle() const
{
	return m_handle;
}

cgpu::DeviceSessionRef cgpu::Queue::getDeviceSession() const
{
	return m_device_session.get();
}

void cgpu::Queue::createSemaphore()
{
	vk::StructureChain<
		vk::SemaphoreCreateInfo,
		vk::SemaphoreTypeCreateInfo>
		chain;

	{
		auto& info = chain.get<vk::SemaphoreCreateInfo>();
		info.flags = {};
	}

	{
		auto& info = chain.get<vk::SemaphoreTypeCreateInfo>();
		info.semaphoreType = vk::SemaphoreType::eTimeline;
		info.initialValue = 0;
	}

	m_semaphore = m_device_session->getHandle().createSemaphore(chain.get(), nullptr, m_device_session->getDispatcher());
}
