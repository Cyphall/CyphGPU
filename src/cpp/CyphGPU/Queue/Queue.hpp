#pragma once

#include <CyphGPU/fwd.hpp>

#include <vulkan/vulkan.hpp>

namespace cgpu
{
class Queue final
{
	class PrivateKey
	{};

public:
	explicit Queue(PrivateKey, DeviceSession& device_session, vk::Queue queue);

	Queue(const Queue&) = delete;
	Queue(Queue&&) = delete;

	Queue& operator=(const Queue&) = delete;
	Queue& operator=(Queue&&) = delete;

	~Queue();

	[[nodiscard]]
	DeviceSessionRef getDeviceSession() const;

	[[nodiscard]]
	const vk::Queue& getHandle() const;

private:
	friend class DeviceSession;

	DeviceSession* m_device_session;

	vk::Queue m_handle;

	vk::Semaphore m_semaphore;

	void createSemaphore();
};
}
