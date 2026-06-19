#include "Swapchain.hpp"

#include <CyphGPU/Device.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/Surface.hpp>
#include <CyphGPU/SwapchainImage.hpp>

#include <flat_set>
#include <ranges>

cgpu::SwapchainPtr cgpu::Swapchain::create(const DeviceSessionPtr& device_session, const SurfacePtr& surface, Desc&& desc)
{
	return std::make_shared<cgpu::Swapchain>(PrivateKey{}, device_session, surface, std::move(desc));
}

cgpu::Swapchain::Swapchain(PrivateKey, const DeviceSessionPtr& device_session, const SurfacePtr& surface, Desc&& desc):
	m_device_session{device_session},
	m_surface{surface},
	m_desc{std::move(desc)}
{
	createSwapchain();
}

cgpu::Swapchain::~Swapchain()
{
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
	swapchain_info.flags = vk::SwapchainCreateFlagBitsKHR::eMutableFormat;
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
			std::make_unique<SwapchainImage>(
				SwapchainImage::PrivateKey{},
				SwapchainImage::Desc{
					.name = std::move(name),
					.format = m_desc.format.format,
					.extent = {extent, 1},
					.usages = m_desc.usages,
					.layers = m_desc.layers,
					.additional_view_formats = m_desc.additional_view_formats,
					.existing_handle = {{
						.image = images[i],
					}},
				},
				*this,
				i
			)
		);
	}
}
