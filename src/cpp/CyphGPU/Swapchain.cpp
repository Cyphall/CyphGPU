#include "Swapchain.hpp"

#include <CyphGPU/Device.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/Image.hpp>
#include <CyphGPU/Queue.hpp>
#include <CyphGPU/Surface.hpp>

#include <flat_set>
#include <ranges>
#include <tracy/Tracy.hpp>

cgpu::SwapchainPtr cgpu::Swapchain::create(const DeviceSessionPtr& device_session, const SurfacePtr& surface, Desc&& desc)
{
	auto swapchain = std::make_shared<cgpu::Swapchain>(PrivateKey{}, device_session, surface, std::move(desc));
	swapchain->performAcquire();
	return swapchain;
}

cgpu::Swapchain::Swapchain(PrivateKey, const DeviceSessionPtr& device_session, const SurfacePtr& surface, Desc&& desc):
	m_device_session{device_session},
	m_surface{surface},
	m_desc{std::move(desc)}
{
	if (!(m_device_session->getDevice()->getCapabilities() & Device::Capability::eSwapchain))
	{
		throw std::logic_error("Cannot create swapchain when device capability Swapchain is not supported.");
	}

	createSwapchain();
	createAcquireSyncObjects();
}

cgpu::Swapchain::~Swapchain()
{
	for (auto& acquire_sync_data : m_acquire_sync_data)
	{
		if (acquire_sync_data.signal_pending)
		{
			std::ignore = m_device_session->getHandle().waitForFences(acquire_sync_data.fence, vk::False, UINT64_MAX, m_device_session->getDispatcher());
		}
		m_device_session->getHandle().destroyFence(acquire_sync_data.fence, nullptr, m_device_session->getDispatcher());
		m_device_session->getHandle().destroySemaphore(acquire_sync_data.semaphore, nullptr, m_device_session->getDispatcher());
	}
	for (auto& image : m_image_data)
	{
		m_device_session->getHandle().destroySemaphore(image.semaphore, nullptr, m_device_session->getDispatcher());
	}
	m_device_session->getHandle().destroySwapchainKHR(m_handle, nullptr, m_device_session->getDispatcher());
}

const cgpu::DeviceSessionPtr& cgpu::Swapchain::getDeviceSession() const
{
	return m_device_session;
}

const cgpu::SurfacePtr& cgpu::Swapchain::getSurface() const
{
	return m_surface;
}

const cgpu::Swapchain::Desc& cgpu::Swapchain::getDesc() const
{
	return m_desc;
}

const vk::SwapchainKHR& cgpu::Swapchain::getHandle()
{
	return m_handle;
}

std::optional<cgpu::ImagePtr> cgpu::Swapchain::tryGetImage()
{
	if (m_status != vk::Result::eSuccess)
	{
		return std::nullopt;
	}

	return {{shared_from_this(), m_image_data[m_acquired_image].image.get()}};
}

void cgpu::Swapchain::presentImage()
{
	ZoneScoped;

	performPresent();

	m_current_frame_index++;

	throttle();

	performAcquire();
}

vk::Semaphore cgpu::Swapchain::createSemaphore()
{
	vk::StructureChain<
		vk::SemaphoreCreateInfo,
		vk::SemaphoreTypeCreateInfo>
		chain;

	auto& create_info = chain.get<vk::SemaphoreCreateInfo>();
	create_info.flags = {};

	auto& type_create_info = chain.get<vk::SemaphoreTypeCreateInfo>();
	type_create_info.semaphoreType = vk::SemaphoreType::eBinary;
	type_create_info.initialValue = 0;

	return m_device_session->getHandle().createSemaphore(chain.get(), nullptr, m_device_session->getDispatcher());
}

vk::Fence cgpu::Swapchain::createFence()
{
	vk::FenceCreateInfo info;
	info.flags = {};

	return m_device_session->getHandle().createFence(info, nullptr, m_device_session->getDispatcher());
}

void cgpu::Swapchain::createSwapchain()
{
	std::flat_set<vk::Format> view_formats_set;
	view_formats_set.emplace(getLinearEquivalent(m_desc.format.format));
	view_formats_set.emplace(getSrgbEquivalent(m_desc.format.format));
	for (vk::Format format : m_desc.additional_view_formats)
	{
		view_formats_set.emplace(getLinearEquivalent(format));
		view_formats_set.emplace(getSrgbEquivalent(format));
	}

	std::vector<vk::Format> view_formats = std::move(view_formats_set).extract();

	std::span<const uint32_t> queue_families = m_device_session->getActiveQueueFamilies();

	vk::PhysicalDeviceSurfaceInfo2KHR surface_info;
	surface_info.surface = m_surface->getHandle();

	vk::SurfaceCapabilities2KHR surface_caps = m_device_session->getDevice()->getHandle().getSurfaceCapabilities2KHR(
		surface_info,
		m_device_session->getDevice()->getContextSession()->getDispatcher()
	);

	glm::uvec2 extent = glm::clamp(
		m_desc.preferred_extent,
		std::bit_cast<glm::uvec2>(surface_caps.surfaceCapabilities.minImageExtent),
		std::bit_cast<glm::uvec2>(surface_caps.surfaceCapabilities.maxImageExtent)
	);

	std::optional<cgpu::SwapchainPtr> old_swapchain;
	if (m_desc.old_swapchain)
	{
		if (auto locked_old_swapchain = m_desc.old_swapchain->lock())
		{
			old_swapchain = locked_old_swapchain;
		}
	}

	vk::StructureChain<
		vk::SwapchainCreateInfoKHR,
		vk::ImageFormatListCreateInfo>
		chain;

	auto& swapchain_info = chain.get<vk::SwapchainCreateInfoKHR>();
	swapchain_info.flags = vk::SwapchainCreateFlagBitsKHR::eMutableFormat |
	                       vk::SwapchainCreateFlagBitsKHR::ePresentId2 |
	                       vk::SwapchainCreateFlagBitsKHR::ePresentWait2;
	swapchain_info.surface = m_surface->getHandle();
	swapchain_info.minImageCount = m_desc.image_count;
	swapchain_info.imageFormat = m_desc.format.format;
	swapchain_info.imageColorSpace = m_desc.format.colorSpace;
	swapchain_info.imageExtent = std::bit_cast<vk::Extent2D>(extent);
	swapchain_info.imageArrayLayers = m_desc.layers;
	swapchain_info.imageUsage = m_desc.usages;
	swapchain_info.imageSharingMode = queue_families.size() > 1 ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive;
	swapchain_info.queueFamilyIndexCount = static_cast<uint32_t>(queue_families.size());
	swapchain_info.pQueueFamilyIndices = queue_families.data();
	swapchain_info.preTransform = surface_caps.surfaceCapabilities.currentTransform;
	swapchain_info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	swapchain_info.presentMode = m_desc.present_mode;
	swapchain_info.clipped = vk::True;
	swapchain_info.oldSwapchain =
		old_swapchain ?
			(*old_swapchain)->getHandle() :
			nullptr;

	auto& image_format_list_info = chain.get<vk::ImageFormatListCreateInfo>();
	image_format_list_info.viewFormatCount = static_cast<uint32_t>(view_formats.size());
	image_format_list_info.pViewFormats = view_formats.data();

	m_handle = m_device_session->getHandle().createSwapchainKHR(swapchain_info, nullptr, m_device_session->getDispatcher());

	std::vector<vk::Image> images = m_device_session->getHandle().getSwapchainImagesKHR(m_handle, m_device_session->getDispatcher());
	m_image_data.reserve(images.size());
	for (uint32_t i = 0; i < images.size(); i++)
	{
		std::string name = std::format("Swapchain image {}", i);
		m_device_session->getHandle().setDebugUtilsObjectNameEXT(images[i], name, m_device_session->getDispatcher());

		m_image_data.emplace_back(
			std::make_unique<Image>(
				Image::PrivateKey{},
				m_device_session,
				Image::Desc{
					.name = std::move(name),
					.format = swapchain_info.imageFormat,
					.extent = {std::bit_cast<glm::uvec2>(swapchain_info.imageExtent), 1},
					.usages = m_desc.usages,
					.layers = swapchain_info.imageArrayLayers,
					.additional_view_formats = m_desc.additional_view_formats,
					.existing_handle = {{
						.image = images[i],
					}},
				}
			),
			createSemaphore()
		);
	}
}

void cgpu::Swapchain::createAcquireSyncObjects()
{
	m_acquire_sync_data.reserve(m_image_data.size());
	for (size_t i = 0; i < m_image_data.size(); i++)
	{
		m_acquire_sync_data.emplace_back(
			createSemaphore(),
			createFence(),
			false
		);
	}
}

void cgpu::Swapchain::performAcquire()
{
	ZoneScoped;

	if (m_status == vk::Result::eErrorOutOfDateKHR)
	{
		return;
	}

	auto& acquire_sync_data = m_acquire_sync_data[m_current_frame_index % m_image_data.size()];

	if (acquire_sync_data.signal_pending)
	{
		ZoneScopedN("Fence wait");

		std::ignore = m_device_session->getHandle().waitForFences(acquire_sync_data.fence, vk::False, UINT64_MAX, m_device_session->getDispatcher());
		m_device_session->getHandle().resetFences(acquire_sync_data.fence, m_device_session->getDispatcher());

		acquire_sync_data.signal_pending = false;
	}

	{
		ZoneScopedN("Acquire");

		vk::AcquireNextImageInfoKHR info;
		info.swapchain = m_handle;
		info.timeout = UINT64_MAX;
		info.semaphore = acquire_sync_data.semaphore;
		info.fence = acquire_sync_data.fence;
		info.deviceMask = 1;

		try
		{
			std::tie(m_status, m_acquired_image) = m_device_session->getHandle().acquireNextImage2KHR(info, m_device_session->getDispatcher());
		}
		catch (const vk::OutOfDateKHRError&)
		{
			m_status = vk::Result::eErrorOutOfDateKHR;
			return;
		}

		acquire_sync_data.signal_pending = true;
	}

	{
		ZoneScopedN("Binary -> Timeline");

		m_image_data[m_acquired_image].image->setSubmitSync(
			m_device_session->getMainQueue()->binaryToSubmitSync(shared_from_this(), acquire_sync_data.semaphore)
		);
	}
}

void cgpu::Swapchain::throttle()
{
	ZoneScoped;

	if (m_status == vk::Result::eErrorOutOfDateKHR)
	{
		return;
	}

	uint64_t max_latency = m_image_data.size() - 1;

	if (m_current_frame_index <= max_latency)
	{
		return;
	}

	vk::PresentWait2InfoKHR info;
	info.presentId = m_current_frame_index - max_latency + 1;
	info.timeout = std::numeric_limits<uint64_t>::max();

	try
	{
		m_status = m_device_session->getHandle().waitForPresent2KHR(m_handle, info, m_device_session->getDispatcher());
	}
	catch (const vk::OutOfDateKHRError&)
	{
		m_status = vk::Result::eErrorOutOfDateKHR;
		return;
	}
}

void cgpu::Swapchain::performPresent()
{
	ZoneScoped;

	if (m_status == vk::Result::eErrorOutOfDateKHR)
	{
		return;
	}

	auto& image_data = m_image_data[m_acquired_image];

	{
		ZoneScopedN("Timeline -> Binary");

		m_device_session->getMainQueue()->submitSyncToBinary(
			shared_from_this(),
			image_data.semaphore,
			*image_data.image->tryGetSubmitSync()
		);
	}

	{
		ZoneScopedN("Present");

		m_status = m_device_session->getMainQueue()->swapchainPresent(
			shared_from_this(),
			m_acquired_image,
			image_data.semaphore,
			m_current_frame_index + 1
		);
	}
}
