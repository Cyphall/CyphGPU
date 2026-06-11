#include "Queue.hpp"

#include <CyphGPU/DeviceSession/DeviceSession.hpp>

cgpu::Queue::Queue(PrivateKey, DeviceSession& device_session, vk::Queue queue):
	m_device_session{&device_session},
	m_handle{queue}
{
	createSemaphore();
}

cgpu::Queue::~Queue()
{
	m_device_session->getHandle().destroySemaphore(m_semaphore, nullptr, m_device_session->getDispatcher());
}

cgpu::DeviceSessionPtr cgpu::Queue::getDeviceSession() const
{
	return m_device_session->shared_from_this();
}

const vk::Queue& cgpu::Queue::getHandle()
{
	return m_handle;
}

void cgpu::Queue::createSemaphore()
{
	vk::StructureChain<
		vk::SemaphoreCreateInfo,
		vk::SemaphoreTypeCreateInfo>
		chain;


	auto& create_info = chain.get<vk::SemaphoreCreateInfo>();
	create_info.flags = {};

	auto& type_create_info = chain.get<vk::SemaphoreTypeCreateInfo>();
	type_create_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_create_info.initialValue = 0;

	m_semaphore = m_device_session->getHandle().createSemaphore(chain.get(), nullptr, m_device_session->getDispatcher());
}
