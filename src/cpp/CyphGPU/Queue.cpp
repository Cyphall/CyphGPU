#include "Queue.hpp"

#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/Swapchain.hpp>

cgpu::Queue::Queue(PrivateKey, DeviceSession& device_session, vk::Queue queue, uint32_t family):
	m_device_session{&device_session},
	m_handle{queue},
	m_family{family}
{
	createSemaphore();
}

cgpu::Queue::~Queue()
{
	waitIdle();

	while (!m_free_fences.empty())
	{
		m_device_session->getHandle().destroyFence(m_free_fences.top(), nullptr, m_device_session->getDispatcher());
		m_free_fences.pop();
	}

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

const uint32_t& cgpu::Queue::getFamily() const
{
	return m_family;
}

void cgpu::Queue::waitIdle()
{
	std::unique_lock lock{m_mutex};

	m_handle.waitIdle(m_device_session->getDispatcher());
	waitAndClearPayloads();
	assert(m_submit_payloads.empty());
	assert(m_present_payloads.empty());
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

cgpu::Queue::SubmitSync cgpu::Queue::binaryToSubmitSync(const SwapchainPtr& swapchain, vk::Semaphore semaphore)
{
	std::unique_lock lock{m_mutex};

	clearCompletedPayloads();

	vk::SemaphoreSubmitInfo wait{
		.semaphore = semaphore,
		.value = 0,
		.stageMask = vk::PipelineStageFlagBits2::eAllCommands,
		.deviceIndex = 0,
	};

	vk::SemaphoreSubmitInfo signal{
		.semaphore = m_semaphore,
		.value = m_next_index++,
		.stageMask = vk::PipelineStageFlagBits2::eAllCommands,
		.deviceIndex = 0,
	};

	vk::SubmitInfo2 info;
	info.flags = {};
	info.waitSemaphoreInfoCount = 1;
	info.pWaitSemaphoreInfos = &wait;
	info.commandBufferInfoCount = 0;
	// info.pCommandBufferInfos;
	info.signalSemaphoreInfoCount = 1;
	info.pSignalSemaphoreInfos = &signal;

	vk::Fence fence = acquireFence();

	m_handle.submit2(info, fence, m_device_session->getDispatcher());

	Payload& payload = m_submit_payloads.emplace();
	payload.objects.emplace_back(swapchain);
	payload.fence = fence;

	return {
		.semaphore = signal.semaphore,
		.value = signal.value,
	};
}

void cgpu::Queue::submitSyncToBinary(const SwapchainPtr& swapchain, vk::Semaphore semaphore, const SubmitSync& submit_sync)
{
	std::unique_lock lock{m_mutex};

	clearCompletedPayloads();

	vk::SemaphoreSubmitInfo wait{
		.semaphore = submit_sync.semaphore,
		.value = submit_sync.value,
		.stageMask = vk::PipelineStageFlagBits2::eAllCommands,
		.deviceIndex = 0,
	};

	vk::SemaphoreSubmitInfo signal{
		.semaphore = semaphore,
		.value = 0,
		.stageMask = vk::PipelineStageFlagBits2::eAllCommands,
		.deviceIndex = 0,
	};

	vk::SubmitInfo2 info;
	info.flags = {};
	info.waitSemaphoreInfoCount = 1;
	info.pWaitSemaphoreInfos = &wait;
	info.commandBufferInfoCount = 0;
	// info.pCommandBufferInfos;
	info.signalSemaphoreInfoCount = 1;
	info.pSignalSemaphoreInfos = &signal;

	vk::Fence fence = acquireFence();

	m_handle.submit2(info, fence, m_device_session->getDispatcher());

	Payload& payload = m_submit_payloads.emplace();
	payload.objects.emplace_back(swapchain);
	payload.fence = fence;
}

vk::Result cgpu::Queue::swapchainPresent(const SwapchainPtr& swapchain, uint32_t index, vk::Semaphore semaphore, uint64_t present_id)
{
	std::unique_lock lock{m_mutex};

	clearCompletedPayloads();

	vk::Fence fence = acquireFence();

	vk::StructureChain<
		vk::PresentInfoKHR,
		vk::SwapchainPresentFenceInfoKHR,
		vk::PresentId2KHR>
		chain;

	auto& present_info = chain.get<vk::PresentInfoKHR>();
	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores = &semaphore;
	present_info.swapchainCount = 1;
	present_info.pSwapchains = &swapchain->getHandle();
	present_info.pImageIndices = &index;
	present_info.pResults = nullptr;

	auto& present_fence_info = chain.get<vk::SwapchainPresentFenceInfoKHR>();
	present_fence_info.swapchainCount = 1;
	present_fence_info.pFences = &fence;

	auto& present_id_info = chain.get<vk::PresentId2KHR>();
	present_id_info.swapchainCount = 1;
	present_id_info.pPresentIds = &present_id;

	vk::Result result{};
	try
	{
		result = m_handle.presentKHR(chain.get(), m_device_session->getDispatcher());
	}
	catch (const vk::OutOfDateKHRError&)
	{
		result = vk::Result::eErrorOutOfDateKHR;
	}

	// Objects are still considered in-use if presentKHR() throws OutOfDate
	Payload& payload = m_present_payloads.emplace();
	payload.objects.emplace_back(swapchain);
	payload.fence = fence;

	return result;
}

vk::Fence cgpu::Queue::acquireFence()
{
	std::unique_lock lock{m_mutex};

	if (!m_free_fences.empty())
	{
		vk::Fence fence = m_free_fences.top();
		m_free_fences.pop();
		return fence;
	}

	vk::FenceCreateInfo info;
	info.flags = {};

	return m_device_session->getHandle().createFence(info, nullptr, m_device_session->getDispatcher());
}

void cgpu::Queue::releaseFence(vk::Fence fence)
{
	std::unique_lock lock{m_mutex};

	m_free_fences.push(fence);
}

void cgpu::Queue::clearCompletedPayloads()
{
	std::unique_lock lock{m_mutex};

	auto clear = [&](std::queue<Payload>& queue) {
		while (!queue.empty() &&
		       m_device_session->getHandle().getFenceStatus(queue.front().fence, m_device_session->getDispatcher()) == vk::Result::eSuccess)
		{
			m_device_session->getHandle().resetFences(queue.front().fence, m_device_session->getDispatcher());
			releaseFence(queue.front().fence);
			queue.pop();
		}
	};

	clear(m_submit_payloads);
	clear(m_present_payloads);
}

void cgpu::Queue::waitAndClearPayloads()
{
	std::unique_lock lock{m_mutex};

	auto clear = [&](std::queue<Payload>& queue) {
		while (!queue.empty())
		{
			std::ignore = m_device_session->getHandle().waitForFences(queue.front().fence, false, UINT64_MAX, m_device_session->getDispatcher());
			m_device_session->getHandle().resetFences(queue.front().fence, m_device_session->getDispatcher());
			releaseFence(queue.front().fence);
			queue.pop();
		}
	};

	clear(m_submit_payloads);
	clear(m_present_payloads);
}
