#pragma once

#include <CyphGPU/fwd.hpp>

#include <flat_map>
#include <mutex>
#include <queue>
#include <stack>
#include <vulkan/vulkan.hpp>

namespace cgpu
{
class CommandRecorder;

class Queue final
{
	class PrivateKey
	{};

public:
	struct Signal
	{
		vk::Semaphore semaphore;
		uint64_t value;
	};

	explicit Queue(PrivateKey, DeviceSession& device_session, vk::Queue queue, uint32_t family);

	Queue(const Queue&) = delete;
	Queue(Queue&&) = delete;

	Queue& operator=(const Queue&) = delete;
	Queue& operator=(Queue&&) = delete;

	~Queue();

	[[nodiscard]]
	DeviceSessionPtr getDeviceSession() const;

	[[nodiscard]]
	const vk::Queue& getHandle();

	[[nodiscard]]
	const uint32_t& getFamily() const;

	void waitIdle();

private:
	friend class DeviceSession;
	friend class Swapchain;
	friend class CommandRecorder;

	struct Payload
	{
		std::vector<std::shared_ptr<void>> objects;
		vk::Fence fence;
	};

	DeviceSession* m_device_session;

	vk::Queue m_handle;

	uint32_t m_family;

	vk::Semaphore m_semaphore{};
	uint64_t m_next_index{1};

	std::queue<Payload> m_submit_payloads{};
	std::queue<Payload> m_present_payloads{};

	std::stack<vk::Fence> m_free_fences{};

	std::recursive_mutex m_mutex{};

	void createSemaphore();

	[[nodiscard]]
	Signal binaryToSignal(
		const SwapchainPtr& swapchain,
		vk::Semaphore semaphore,
		vk::CommandBuffer cmdbuf
	);

	[[nodiscard]]
	Signal signalToBinary(
		const SwapchainPtr& swapchain,
		vk::Semaphore semaphore,
		vk::CommandBuffer cmdbuf,
		std::span<const vk::Semaphore> wait_semaphores,
		std::span<const uint64_t> wait_values
	);

	[[nodiscard]]
	vk::Result swapchainPresent(const SwapchainPtr& swapchain, uint32_t index, vk::Semaphore semaphore, uint64_t present_id);

	[[nodiscard]]
	vk::Fence acquireFence();
	void releaseFence(vk::Fence fence);

	void clearCompletedPayloads();
	void waitAndClearPayloads();

	Signal submit(
		std::span<const vk::CommandBuffer> cmdbufs,
		std::span<const vk::Semaphore> wait_semaphores,
		std::span<const uint64_t> wait_values,
		std::vector<std::shared_ptr<void>>&& referenced_objects
	);
};
}
