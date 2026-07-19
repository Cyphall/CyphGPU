#pragma once

#include <CyphGPU/fwd.hpp>
#include <CyphGPU/MemoryType.hpp>
#include <CyphGPU/Resource.hpp>
#include <CyphGPU/ShaderTypes.hpp>
#include <CyphGPU/Utils.hpp>

#include <flat_map>
#include <glm/glm.hpp>
#include <optional>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace cgpu
{
class CommandRecorder;

class Image final : public Resource
{
	class PrivateKey
	{};

public:
	struct Desc
	{
		struct ExistingHandle
		{
			vk::Image image;
		};

		// Required
		std::string name;
		vk::Format format;
		glm::uvec3 extent;
		vk::ImageUsageFlags usages; //TODO: Use vk::ImageUsageFlags2 when VK_KHR_extended_flags becomes more widely supported.

		// Optional
		vk::ImageType type{vk::ImageType::e2D};
		MemoryType memory_type{MemoryType::eGPUHighPrio};
		uint32_t levels{1};
		uint32_t layers{1};
		vk::SampleCountFlagBits samples{vk::SampleCountFlagBits::e1};
		std::vector<vk::Format> additional_view_formats{};
		/// For 2D images, must have extent.x == extent.y, layers >= 6 and samples == 1.
		bool allow_cube_view{false};
		/// For 3D images.
		bool allow_2d_array_view{false};
		/// For compressed images.
		bool allow_block_texel_view{false};
		/// Other desc parameters must still match what the image was created with.
		std::optional<ExistingHandle> existing_handle{};
	};

	struct SampledDescriptorOverrides
	{
		/// Default: Same as image type.
		std::optional<vk::ImageViewType> type;
		/// Default: Image format.
		std::optional<vk::Format> format;
		/// Default: All levels.
		std::optional<Range<uint32_t>> levels;
		/// Default: First layer = 0. Layer count = 6 if type is Cube, all layers if type is *Array, 1 otherwise.
		std::optional<Range<uint32_t>> layers;
		/// Default: Main aspect. For multi-aspect formats, this field must be set.
		std::optional<vk::ImageAspectFlagBits> aspect;
		/// Default: Identity.
		std::optional<vk::ComponentMapping> swizzle;
	};

	struct StorageDescriptorOverrides
	{
		/// Default: Same as image type.
		std::optional<vk::ImageViewType> type;
		/// Default: Image format.
		///
		/// Storage descriptors do not support sRGB formats.
		/// If the format is an sRGB format, the equivalent linear format will be used instead.
		std::optional<vk::Format> format;
		/// Default: 0.
		std::optional<uint32_t> level;
		/// Default: First layer = 0. Layer count = 6 if type is Cube, all layers if type is *Array, 1 otherwise.
		std::optional<Range<uint32_t>> layers;
		/// Default: Main aspect. For multi-aspect formats, this field must be set.
		std::optional<vk::ImageAspectFlagBits> aspect;
	};

	[[nodiscard]]
	static ImagePtr create(const DeviceSessionPtr& device_session, Desc&& desc);

	explicit Image(PrivateKey, const DeviceSessionPtr& device_session, Desc&& desc);

	Image(const Image&) = delete;
	Image(Image&&) = delete;

	Image& operator=(const Image&) = delete;
	Image& operator=(Image&&) = delete;

	~Image() override;

	[[nodiscard]]
	const DeviceSessionPtr& getDeviceSession() const;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::Image& getHandle();

	[[nodiscard]]
	SampledImageHandle getSampledDescriptorIndirect(const SampledDescriptorOverrides& overrides = {});

	[[nodiscard]]
	StorageImageHandle getStorageDescriptorIndirect(const StorageDescriptorOverrides& overrides = {});

private:
	friend class Swapchain;
	friend class CommandRecorder;

	struct SampledDescriptorInfo
	{
		vk::ImageViewType type{};
		vk::Format format{};
		Range<uint32_t> levels{};
		Range<uint32_t> layers{};
		vk::ImageAspectFlagBits aspect{};
		vk::ComponentMapping swizzle{};

		auto operator<=>(const SampledDescriptorInfo&) const = default;
	};

	struct StorageDescriptorInfo
	{
		vk::ImageViewType type{};
		vk::Format format{};
		uint32_t level{};
		Range<uint32_t> layers{};
		vk::ImageAspectFlagBits aspect{};

		auto operator<=>(const StorageDescriptorInfo&) const = default;
	};

	struct AttachmentViewInfo
	{
		vk::Format format{};
		uint32_t level{};
		Range<uint32_t> layers{};
		vk::ImageAspectFlags aspects{};
		vk::ImageUsageFlagBits usage{};

		auto operator<=>(const AttachmentViewInfo&) const = default;
	};

	DeviceSessionPtr m_device_session;

	Desc m_desc;

	vk::Image m_handle{};
	std::optional<VmaAllocation> m_alloc{};

	vk::ImageViewType m_default_view_type{};
	std::optional<vk::ImageAspectFlagBits> m_default_view_aspect{};

	std::flat_map<SampledDescriptorInfo, uint32_t> m_sampled_cache;
	std::flat_map<StorageDescriptorInfo, uint32_t> m_storage_cache;

	//TODO: remove once we have an extension to remove image views from attachments
	std::flat_map<AttachmentViewInfo, vk::ImageView> m_attachment_cache;

	void createImage();

	uint32_t calcDefaultLayerCount(vk::ImageViewType type);

	[[nodiscard]]
	vk::ImageView getAttachmentView(vk::Format format, uint32_t level, Range<uint32_t> layers, vk::ImageAspectFlags aspects, vk::ImageUsageFlagBits usage);
};
}
