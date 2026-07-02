#pragma once

#include <CyphGPU/fwd.hpp>

#include <mutex>
#include <queue>
#include <stack>
#include <vulkan/vulkan.hpp>

namespace cgpu
{
class Queue final
{
	class PrivateKey
	{};

public:
	struct SubmitSync
	{
		vk::Semaphore semaphore;
		uint64_t value;
	};

	explicit Queue(PrivateKey, DeviceSession& device_session, vk::Queue queue);

	Queue(const Queue&) = delete;
	Queue(Queue&&) = delete;

	Queue& operator=(const Queue&) = delete;
	Queue& operator=(Queue&&) = delete;

	~Queue();

	[[nodiscard]]
	DeviceSessionPtr getDeviceSession() const;

	[[nodiscard]]
	const vk::Queue& getHandle();

	void waitIdle();

private:
	friend class DeviceSession;
	friend class Swapchain;

	struct Payload
	{
		std::vector<std::shared_ptr<void>> objects;
		vk::Fence fence;
	};

	DeviceSession* m_device_session;

	vk::Queue m_handle;

	vk::Semaphore m_semaphore{};
	uint64_t m_next_index{1};

	std::queue<Payload> m_submit_payloads{};
	std::queue<Payload> m_present_payloads{};

	std::stack<vk::Fence> m_free_fences{};

	std::recursive_mutex m_mutex{};

	void createSemaphore();

	SubmitSync binaryToSubmitSync(const SwapchainPtr& swapchain, vk::Semaphore semaphore);
	void submitSyncToBinary(const SwapchainPtr& swapchain, vk::Semaphore semaphore, const SubmitSync& submit_sync);

	vk::Result swapchainPresent(const SwapchainPtr& swapchain, uint32_t index, vk::Semaphore semaphore, uint64_t present_id);

	[[nodiscard]]
	vk::Fence acquireFence();
	void releaseFence(vk::Fence fence);

	void clearCompletedPayloads();
	void waitAndClearPayloads();
};
}
