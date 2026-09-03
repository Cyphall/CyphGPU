#include "Queue.hpp"

#include <CyphGPU/ContextSession.hpp>
#include <CyphGPU/detail/BumpAllocator.hpp>
#include <CyphGPU/Device.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/Swapchain.hpp>

#include <boost/container/small_vector.hpp>
#include <tracy/Tracy.hpp>

cgpu::Queue::Queue(PrivateKey, DeviceSession& device_session, vk::Queue queue, uint32_t family, vk::QueueFlags caps, std::string_view name):
	m_device_session{&device_session},
	m_handle{queue},
	m_family{family},
	m_caps{caps}
{
	m_device_session->getHandle().setDebugUtilsObjectNameEXT(queue, std::string{name}, m_device_session->getDispatcher());

	createSemaphore();
#if defined(TRACY_ENABLE)
	createTracyContext(name);
#endif
}

cgpu::Queue::~Queue()
{
	waitIdle();

#if defined(TRACY_ENABLE)
	TracyVkDestroy(m_tracy_context);
#endif

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

const vk::QueueFlags& cgpu::Queue::getCapabilities() const
{
	return m_caps;
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

#if defined(TRACY_ENABLE)
void cgpu::Queue::createTracyContext(std::string_view name)
{
	m_tracy_context = TracyVkContextHostCalibrated(
		m_device_session->getDevice()->getContextSession()->getHandle(),
		m_device_session->getDevice()->getHandle(),
		m_device_session->getHandle(),
		m_device_session->getDevice()->getContextSession()->getDispatcher().vkGetInstanceProcAddr,
		m_device_session->getDispatcher().vkGetDeviceProcAddr
	);

	TracyVkContextName(m_tracy_context, name.data(), static_cast<uint16_t>(name.size()));
}
#endif

void cgpu::Queue::timelineToBinary(
	const SwapchainPtr& swapchain,
	vk::Semaphore semaphore,
	vk::CommandBuffer cmd_buf,
	std::span<const vk::Semaphore> wait_semaphores,
	std::span<const uint64_t> wait_values
)
{
	std::unique_lock lock{m_mutex};

	clearCompletedPayloads();

	std::vector<vk::SemaphoreSubmitInfo> wait_infos;
	wait_infos.resize(wait_semaphores.size());
	for (size_t i = 0; i < wait_infos.size(); i++)
	{
		wait_infos[i] = vk::SemaphoreSubmitInfo{
			.semaphore = wait_semaphores[i],
			.value = wait_values[i],
			.stageMask = vk::PipelineStageFlagBits2::eAllCommands,
			.deviceIndex = 0,
		};
	}

	std::array cmd_buf_infos{
		vk::CommandBufferSubmitInfo{
			.commandBuffer = cmd_buf,
			.deviceMask = 1,
		},
	};

	std::array signal_infos{
		vk::SemaphoreSubmitInfo{
			.semaphore = m_semaphore,
			.value = m_next_index++,
			.stageMask = vk::PipelineStageFlagBits2::eAllCommands,
			.deviceIndex = 0,
		},
		vk::SemaphoreSubmitInfo{
			.semaphore = semaphore,
			.value = 0,
			.stageMask = vk::PipelineStageFlagBits2::eAllCommands,
			.deviceIndex = 0,
		},
	};

	vk::SubmitInfo2 info;
	info.flags = {};
	info.waitSemaphoreInfoCount = static_cast<uint32_t>(wait_infos.size());
	info.pWaitSemaphoreInfos = wait_infos.data();
	info.commandBufferInfoCount = static_cast<uint32_t>(cmd_buf_infos.size());
	info.pCommandBufferInfos = cmd_buf_infos.data();
	info.signalSemaphoreInfoCount = static_cast<uint32_t>(signal_infos.size());
	info.pSignalSemaphoreInfos = signal_infos.data();

	m_handle.submit2(info, nullptr, m_device_session->getDispatcher());

	SubmitPayload& payload = m_submit_payloads.emplace_back();
	payload.objects.emplace_back(swapchain);
	payload.semaphore_value = signal_infos[0].value;
}

vk::Result cgpu::Queue::swapchainPresent(const SwapchainPtr& swapchain, uint32_t index, vk::Semaphore semaphore)
{
	std::unique_lock lock{m_mutex};

	clearCompletedPayloads();

	vk::Fence fence = acquireFence();

	vk::StructureChain<
		vk::PresentInfoKHR,
		vk::SwapchainPresentFenceInfoKHR>
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
	PresentPayload& payload = m_present_payloads.emplace_back();
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

void cgpu::Queue::releaseFences(std::span<const vk::Fence> fences)
{
	std::unique_lock lock{m_mutex};

	m_free_fences.push_range(fences);
}

void cgpu::Queue::clearCompletedPayloads()
{
	std::unique_lock lock{m_mutex};

	while (!m_submit_payloads.empty())
	{
		vk::SemaphoreWaitInfo info;
		info.flags = {};
		info.semaphoreCount = 1;
		info.pSemaphores = &m_semaphore;
		info.pValues = &m_submit_payloads.front().semaphore_value;

		if (m_device_session->getHandle().waitSemaphores(info, 0, m_device_session->getDispatcher()) != vk::Result::eSuccess)
		{
			break;
		}

		m_submit_payloads.pop_front();
	}

	while (!m_present_payloads.empty())
	{
		vk::Fence fence = m_present_payloads.front().fence;

		if (m_device_session->getHandle().waitForFences(fence, vk::True, 0, m_device_session->getDispatcher()) != vk::Result::eSuccess)
		{
			break;
		}

		m_device_session->getHandle().resetFences(fence, m_device_session->getDispatcher());
		releaseFences({{fence}});

		m_present_payloads.pop_front();
	}
}

void cgpu::Queue::waitAndClearPayloads()
{
	std::unique_lock lock{m_mutex};

	if (!m_submit_payloads.empty())
	{
		vk::SemaphoreWaitInfo info;
		info.flags = {};
		info.semaphoreCount = 1;
		info.pSemaphores = &m_semaphore;
		info.pValues = &m_submit_payloads.back().semaphore_value;

		std::ignore = m_device_session->getHandle().waitSemaphores(info, std::numeric_limits<uint64_t>::max(), m_device_session->getDispatcher());

		m_submit_payloads.clear();
	}

	if (!m_present_payloads.empty())
	{
		boost::container::small_vector<vk::Fence, 8> fences;
		fences.reserve(m_present_payloads.size());
		for (auto& payload : m_present_payloads)
		{
			fences.emplace_back(payload.fence);
		}

		std::ignore = m_device_session->getHandle().waitForFences(fences, vk::True, std::numeric_limits<uint64_t>::max(), m_device_session->getDispatcher());

		m_device_session->getHandle().resetFences(fences, m_device_session->getDispatcher());
		releaseFences(fences);

		m_present_payloads.clear();
	}
}

cgpu::Queue::Signal cgpu::Queue::submit(
	detail::BumpMemoryResource& bump_memory,
	std::span<const vk::CommandBuffer> cmd_bufs,
	std::span<const vk::Semaphore> wait_semaphores,
	std::span<const uint64_t> wait_values,
	std::vector<std::shared_ptr<void>>&& referenced_objects
)
{
	ZoneScoped;

	std::unique_lock lock{m_mutex};

	clearCompletedPayloads();

	detail::BumpVector<vk::SemaphoreSubmitInfo> wait_infos{detail::BumpAllocator{bump_memory}};
	wait_infos.resize(wait_semaphores.size());
	for (size_t i = 0; i < wait_infos.size(); i++)
	{
		wait_infos[i] = vk::SemaphoreSubmitInfo{
			.semaphore = wait_semaphores[i],
			.value = wait_values[i],
			.stageMask = vk::PipelineStageFlagBits2::eAllCommands,
			.deviceIndex = 0,
		};
	}

	detail::BumpVector<vk::CommandBufferSubmitInfo> cmd_buf_infos{detail::BumpAllocator{bump_memory}};
	cmd_buf_infos.reserve(cmd_bufs.size());
	for (vk::CommandBuffer cmd_buf : cmd_bufs)
	{
		auto& cmd_buf_info = cmd_buf_infos.emplace_back();
		cmd_buf_info.commandBuffer = cmd_buf;
		cmd_buf_info.deviceMask = 1;
	}

	std::array signal_infos{
		vk::SemaphoreSubmitInfo{
			.semaphore = m_semaphore,
			.value = m_next_index++,
			.stageMask = vk::PipelineStageFlagBits2::eAllCommands,
			.deviceIndex = 0,
		},
	};

	vk::SubmitInfo2 info;
	info.flags = {};
	info.waitSemaphoreInfoCount = static_cast<uint32_t>(wait_infos.size());
	info.pWaitSemaphoreInfos = wait_infos.data();
	info.commandBufferInfoCount = static_cast<uint32_t>(cmd_buf_infos.size());
	info.pCommandBufferInfos = cmd_buf_infos.data();
	info.signalSemaphoreInfoCount = static_cast<uint32_t>(signal_infos.size());
	info.pSignalSemaphoreInfos = signal_infos.data();

	m_handle.submit2(info, nullptr, m_device_session->getDispatcher());

	SubmitPayload& payload = m_submit_payloads.emplace_back();
	payload.objects = std::move(referenced_objects);
	payload.semaphore_value = signal_infos[0].value;

	return {
		.semaphore = signal_infos[0].semaphore,
		.value = signal_infos[0].value,
	};
}

#if defined(TRACY_ENABLE)
TracyVkCtx cgpu::Queue::getTracyContext()
{
	return m_tracy_context;
}

void cgpu::Queue::tracyCollect()
{
	TracyVkCollectHost(m_tracy_context);
}
#endif
