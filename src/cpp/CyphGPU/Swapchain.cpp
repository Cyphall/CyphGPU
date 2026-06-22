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

	if (!swapchain->performAcquire())
	{
		throw std::runtime_error("Failed to acquire initial swapchain image.");
	}

	return swapchain;
}

cgpu::Swapchain::Swapchain(PrivateKey, const DeviceSessionPtr& device_session, const SurfacePtr& surface, Desc&& desc):
	m_device_session{device_session},
	m_surface{surface},
	m_desc{std::move(desc)}
{
	createSwapchain();
	createSyncObjects();
}

cgpu::Swapchain::~Swapchain()
{
	for (size_t i = 0; i < m_images.size(); i++)
	{
		std::ignore = m_device_session->getHandle().waitForFences(m_acquire_fences[i], vk::False, UINT64_MAX, m_device_session->getDispatcher());
		m_device_session->getHandle().destroyFence(m_acquire_fences[i], nullptr, m_device_session->getDispatcher());
		m_device_session->getHandle().destroySemaphore(m_acquire_semahores[i], nullptr, m_device_session->getDispatcher());
		m_device_session->getHandle().destroySemaphore(m_present_semahores[i], nullptr, m_device_session->getDispatcher());
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

cgpu::ImagePtr cgpu::Swapchain::getImage()
{
	return {shared_from_this(), m_images[m_acquired_image].get()};
}

bool cgpu::Swapchain::presentImage()
{
	ZoneScoped;

	if (!performPresent())
	{
		return false;
	}

	throttle();

	m_current_frame_index++;

	if (!performAcquire())
	{
		return false;
	}

	return true;
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
	swapchain_info.imageSharingMode = vk::SharingMode::eConcurrent;
	swapchain_info.queueFamilyIndexCount = static_cast<uint32_t>(queue_families.size());
	swapchain_info.pQueueFamilyIndices = queue_families.data();
	swapchain_info.preTransform = surface_caps.surfaceCapabilities.currentTransform;
	swapchain_info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	swapchain_info.presentMode = m_desc.present_mode;
	swapchain_info.clipped = vk::True;
	swapchain_info.oldSwapchain =
		m_desc.old_swapchain ?
			m_desc.old_swapchain->lock()->getHandle() :
			nullptr;

	auto& image_format_list_info = chain.get<vk::ImageFormatListCreateInfo>();
	image_format_list_info.viewFormatCount = static_cast<uint32_t>(view_formats.size());
	image_format_list_info.pViewFormats = view_formats.data();

	m_handle = m_device_session->getHandle().createSwapchainKHR(swapchain_info, nullptr, m_device_session->getDispatcher());

	std::vector<vk::Image> images = m_device_session->getHandle().getSwapchainImagesKHR(m_handle, m_device_session->getDispatcher());
	m_images.reserve(images.size());
	for (uint32_t i = 0; i < images.size(); i++)
	{
		std::string name = std::format("Swapchain image {}", i);
		m_device_session->getHandle().setDebugUtilsObjectNameEXT(images[i], name, m_device_session->getDispatcher());

		m_images.emplace_back(
			std::make_unique<Image>(
				Image::PrivateKey{},
				m_device_session,
				Image::Desc{
					.name = std::move(name),
					.format = m_desc.format.format,
					.extent = {extent, 1},
					.usages = m_desc.usages,
					.layers = m_desc.layers,
					.additional_view_formats = m_desc.additional_view_formats,
					.existing_handle = {{
						.image = images[i],
					}},
				}
			)
		);
	}
}

void cgpu::Swapchain::createSyncObjects()
{
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

		m_acquire_semahores.reserve(m_images.size());
		m_present_semahores.reserve(m_images.size());
		for (size_t i = 0; i < m_images.size(); i++)
		{
			m_acquire_semahores.emplace_back(m_device_session->getHandle().createSemaphore(chain.get(), nullptr, m_device_session->getDispatcher()));
			m_present_semahores.emplace_back(m_device_session->getHandle().createSemaphore(chain.get(), nullptr, m_device_session->getDispatcher()));
		}
	}

	{
		vk::FenceCreateInfo info;
		info.flags = vk::FenceCreateFlagBits::eSignaled;

		m_acquire_fences.reserve(m_images.size());
		for (size_t i = 0; i < m_images.size(); i++)
		{
			m_acquire_fences.emplace_back(m_device_session->getHandle().createFence(info, nullptr, m_device_session->getDispatcher()));
		}
	}
}

bool cgpu::Swapchain::performAcquire()
{
	ZoneScoped;

	vk::Semaphore semaphore = m_acquire_semahores[m_current_frame_index % m_images.size()];
	vk::Fence fence = m_acquire_fences[m_current_frame_index % m_images.size()];

	{
		ZoneScopedN("Fence wait");

		std::ignore = m_device_session->getHandle().waitForFences(fence, vk::False, UINT64_MAX, m_device_session->getDispatcher());
		m_device_session->getHandle().resetFences(fence, m_device_session->getDispatcher());
	}

	{
		ZoneScopedN("Acquire");

		vk::AcquireNextImageInfoKHR info;
		info.swapchain = m_handle;
		info.timeout = UINT64_MAX;
		info.semaphore = semaphore;
		info.fence = fence;
		info.deviceMask = 1;

		try
		{
			vk::Result result{};
			std::tie(result, m_acquired_image) = m_device_session->getHandle().acquireNextImage2KHR(info, m_device_session->getDispatcher());
			if (result == vk::Result::eSuboptimalKHR)
			{
				return false;
			}
		}
		catch (const vk::OutOfDateKHRError&)
		{
			return false;
		}
	}

	{
		ZoneScopedN("Binary -> Timeline");

		m_images[m_acquired_image]->setSubmitSync(m_device_session->getMainQueue()->binaryToSubmitSync(shared_from_this(), semaphore));
	}

	return true;
}

void cgpu::Swapchain::throttle()
{
	ZoneScoped;

	uint64_t max_latency = m_desc.image_count - 1;

	if (m_current_frame_index < max_latency)
	{
		return;
	}

	vk::PresentWait2InfoKHR info;
	info.presentId = m_current_frame_index - max_latency;
	info.timeout = std::numeric_limits<uint64_t>::max();

	std::ignore = m_device_session->getHandle().waitForPresent2KHR(m_handle, info, m_device_session->getDispatcher());
}

bool cgpu::Swapchain::performPresent()
{
	ZoneScoped;

	{
		ZoneScopedN("Timeline -> Binary");

		m_device_session->getMainQueue()->submitSyncToBinary(
			shared_from_this(),
			m_present_semahores[m_acquired_image],
			*m_images[m_acquired_image]->tryGetSubmitSync()
		);
	}

	{
		ZoneScopedN("Present");

		return m_device_session->getMainQueue()->swapchainPresent(
			shared_from_this(),
			m_acquired_image,
			m_present_semahores[m_acquired_image],
			m_current_frame_index
		);
	}
}
