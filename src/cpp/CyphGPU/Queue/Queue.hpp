#pragma once

#include <CyphGPU/DependencyObject.hpp>
#include <CyphGPU/fwd.hpp>

#include <vulkan/vulkan.hpp>

namespace cgpu
{
class Queue final : public DependencyObject<Queue>
{
	class PrivateKey
	{};

public:
	explicit Queue(PrivateKey, DeviceSession& device_session, vk::Queue queue);

	~Queue() override;

	[[nodiscard]]
	DeviceSessionRef getDeviceSession() const;

	[[nodiscard]]
	const vk::Queue& getHandle() const;

private:
	friend class DeviceSession;

	DependencyParent<DeviceSession> m_device_session;

	vk::Queue m_handle;

	vk::Semaphore m_semaphore;

	void createSemaphore();
};
}
