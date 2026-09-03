#pragma once

#include <CyphGPU/fwd.hpp>

#include <mutex>
#include <queue>
#include <stack>
#include <vulkan/vulkan.hpp>

#if defined(TRACY_ENABLE)
#	include <tracy/TracyVulkan.hpp>
#endif

namespace cgpu
{
namespace detail
{
class BumpMemoryResource;
}

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

	explicit Queue(PrivateKey, DeviceSession& device_session, vk::Queue queue, uint32_t family, vk::QueueFlags caps, std::string_view name);

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

	[[nodiscard]]
	const vk::QueueFlags& getCapabilities() const;

	void waitIdle();

private:
	friend class DeviceSession;
	friend class Swapchain;
	friend class CommandRecorder;

	struct SubmitPayload
	{
		std::vector<std::shared_ptr<void>> objects;
		uint64_t semaphore_value;
	};

	struct PresentPayload
	{
		std::vector<std::shared_ptr<void>> objects;
		vk::Fence fence;
	};

	DeviceSession* m_device_session;

	vk::Queue m_handle;

	uint32_t m_family;
	vk::QueueFlags m_caps;

	vk::Semaphore m_semaphore{};
	uint64_t m_next_index{1};

	std::deque<SubmitPayload> m_submit_payloads{};
	std::deque<PresentPayload> m_present_payloads{};

	std::stack<vk::Fence> m_free_fences{};

	std::recursive_mutex m_mutex{};

#if defined(TRACY_ENABLE)
	TracyVkCtx m_tracy_context{};
#endif

	void createSemaphore();
#if defined(TRACY_ENABLE)
	void createTracyContext(std::string_view name);
#endif

	[[nodiscard]]
	Signal timelineToBinary(
		const SwapchainPtr& swapchain,
		vk::Semaphore semaphore,
		vk::CommandBuffer cmd_buf,
		std::span<const vk::Semaphore> wait_semaphores,
		std::span<const uint64_t> wait_values
	);

	[[nodiscard]]
	vk::Result swapchainPresent(const SwapchainPtr& swapchain, uint32_t index, vk::Semaphore semaphore);

	[[nodiscard]]
	vk::Fence acquireFence();
	void releaseFences(std::span<const vk::Fence> fences);

	void clearCompletedPayloads();
	void waitAndClearPayloads();

	Signal submit(
		detail::BumpMemoryResource& bump_memory,
		std::span<const vk::CommandBuffer> cmd_bufs,
		std::span<const vk::Semaphore> wait_semaphores,
		std::span<const uint64_t> wait_values,
		std::vector<std::shared_ptr<void>>&& referenced_objects
	);

#if defined(TRACY_ENABLE)
	TracyVkCtx getTracyContext();
	void tracyCollect();
#endif
};
}
