#pragma once

#include <CyphGPU/fwd.hpp>
#include <CyphGPU/Utils.hpp>

#include <glm/glm.hpp>
#include <optional>
#include <vulkan/vulkan.hpp>

namespace cgpu
{
class Swapchain final : public std::enable_shared_from_this<Swapchain>
{
	class PrivateKey
	{};

public:
	struct Desc
	{
		/// Required.
		vk::SurfaceFormatKHR format CGPU_REQUIRED;
		/// Required.
		glm::uvec2 preferred_extent CGPU_REQUIRED;
		/// Required.
		vk::ImageUsageFlags usages CGPU_REQUIRED;
		/// Optional. Default: 1.
		uint32_t preferred_layers{1};
		/// Optional. Default: No additional view format.
		std::vector<vk::Format> additional_view_formats{};
		/// Optional. Default: FIFO.
		vk::PresentModeKHR present_mode{vk::PresentModeKHR::eFifo};
		/// Optional. Default: 3.
		uint32_t preferred_image_count{3};
		/// Optional. Default: Opaque.
		vk::CompositeAlphaFlagBitsKHR alpha_mode{vk::CompositeAlphaFlagBitsKHR::eOpaque};
		/// Optional. Default: No old swapchain.
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
	const glm::uvec2& getExtent() const;

	[[nodiscard]]
	const uint32_t& getLayers() const;

	[[nodiscard]]
	uint32_t getImageCount() const;

	[[nodiscard]]
	const vk::SwapchainKHR& getHandle();

	/// If std::nullopt is returned, the swapchain has become out-of-date or suboptimal and must be recreated.
	///
	/// Images other than the current image must not be accessed.
	[[nodiscard]]
	std::optional<ImagePtr> tryGetImage();

	void presentImage();

private:
	struct ImageData
	{
		std::unique_ptr<Image> image;
		vk::Semaphore semaphore{};
	};

	DeviceSessionPtr m_device_session;
	SurfacePtr m_surface;

	Desc m_desc;

	glm::uvec2 m_extent{};
	uint32_t m_layers{};

	vk::SwapchainKHR m_handle{};
	std::vector<ImageData> m_image_data{};

	vk::Fence m_acquire_fence{};

	uint32_t m_acquired_image{};
	vk::Result m_status{vk::Result::eSuccess};

	vk::CommandPool m_layout_change_cmdpool{};
	std::vector<vk::CommandBuffer> m_present_layout_change_cmd_bufs{};

	void createSwapchain();
	void createAcquireFence();
	void createLayoutChangeObjects();

	void performAcquire();
	void performPresent();
};
}
