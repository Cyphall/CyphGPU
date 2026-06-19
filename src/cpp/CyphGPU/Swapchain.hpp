#pragma once

#include <CyphGPU/fwd.hpp>
#include <CyphGPU/Utils.hpp>

#include <glm/glm.hpp>
#include <optional>
#include <vulkan/vulkan.hpp>

namespace cgpu
{
class Swapchain final
{
	class PrivateKey
	{};

public:
	struct Desc
	{
		// Required
		vk::SurfaceFormatKHR format;
		glm::uvec2 preferred_extent;
		vk::ImageUsageFlags usages;

		// Optional
		uint32_t layers{1};
		std::vector<vk::Format> additional_view_formats{};
		vk::PresentModeKHR present_mode{vk::PresentModeKHR::eFifo};
		uint32_t image_count{3};
		std::optional<std::weak_ptr<Swapchain>> old_swapchain{};
	};

	[[nodiscard]]
	static SwapchainPtr create(const DeviceSessionPtr& device_session, const SurfacePtr& surface, Desc&& desc);

	explicit Swapchain(PrivateKey, const DeviceSessionPtr& device_session, const SurfacePtr& surface, Desc&& desc);

	Swapchain(const Swapchain&) = delete;
	Swapchain(Swapchain&&) = delete;

	Swapchain& operator=(const Swapchain&) = delete;
	Swapchain& operator=(Swapchain&&) = delete;

	~Swapchain();

	[[nodiscard]]
	const DeviceSessionPtr& getDeviceSession() const;

	[[nodiscard]]
	const SurfacePtr& getSurface() const;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::SwapchainKHR& getHandle();

private:
	DeviceSessionPtr m_device_session;
	SurfacePtr m_surface;

	Desc m_desc;

	vk::SwapchainKHR m_handle{};

	void createSwapchain();
};
}
