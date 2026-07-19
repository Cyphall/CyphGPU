#pragma once

#include <CyphGPU/fwd.hpp>

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
		// Required
		vk::SurfaceFormatKHR format;
		glm::uvec2 preferred_extent;
		vk::ImageUsageFlags usages;

		// Optional
		uint32_t preferred_layers{1};
		std::vector<vk::Format> additional_view_formats{};
		vk::PresentModeKHR present_mode{vk::PresentModeKHR::eFifo};
		uint32_t preferred_image_count{3};
		vk::CompositeAlphaFlagBitsKHR alpha_mode{vk::CompositeAlphaFlagBitsKHR::eOpaque};
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

	// If std::nullopt is returned, the swapchain has become out-of-date or suboptimal and must be recreated
	[[nodiscard]]
	std::optional<ImagePtr> tryGetImage();

	void presentImage();

private:
	struct AcquireSyncData
	{
		vk::Semaphore semaphore{};
		vk::Fence fence{};
		bool signal_pending{};
	};

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

	std::vector<AcquireSyncData> m_acquire_sync_data{};

	uint64_t m_current_frame_index{0};
	uint32_t m_acquired_image{};
	vk::Result m_status{vk::Result::eSuccess};

	vk::CommandPool m_layout_change_cmdpool{};
	std::vector<vk::CommandBuffer> m_acquire_layout_change_cmdbufs{};
	std::vector<vk::CommandBuffer> m_present_layout_change_cmdbufs{};

	vk::Semaphore createSemaphore();
	vk::Fence createFence();

	void createSwapchain();
	void createAcquireSyncObjects();
	void createLayoutChangeObjects();

	void performAcquire();
	void throttle();
	void performPresent();
};
}
