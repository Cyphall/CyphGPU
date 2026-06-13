#pragma once

#include <CyphGPU/fwd.hpp>
#include <CyphGPU/MemoryType.hpp>
#include <CyphGPU/Utils.hpp>

#include <flat_map>
#include <glm/glm.hpp>
#include <optional>
#include <vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>

namespace cgpu
{
class Image final
{
	class PrivateKey
	{};

public:
	struct Desc
	{
		// Required
		std::string name;
		vk::Format format; /// Do not use an sRGB format here. sRGB conversion is controlled per-descriptor/attachment.
		glm::uvec3 extent;
		vk::ImageUsageFlags usages; //TODO: Use vk::ImageUsageFlags2 when VK_KHR_extended_flags becomes more widely supported.

		// Optional
		MemoryType memory_type{MemoryType::eGPUHighPrio};
		uint32_t levels{1};
		uint32_t layers{1};
		vk::SampleCountFlagBits samples{vk::SampleCountFlagBits::e1};
		std::vector<vk::Format> additional_view_formats{};
		bool allow_cube_view{false}; /// For 2D images, must have extent.x == extent.y, layers >= 6 and samples == 1.
		bool allow_2d_array_view{false}; /// For 3D images.
		bool allow_block_texel_view{false}; /// For 3D images.
	};

	struct SampledDescriptorOverrides
	{
		std::optional<vk::ImageViewType> type; /// Default: 1D/2D/3D depending on image extent.
		std::optional<vk::Format> format; /// Default: Image format. Do not use an sRGB format here. Instead, set srgb_conversion.
		std::optional<Range<uint32_t>> levels; /// Default: All levels.
		std::optional<Range<uint32_t>> layers; /// Default: First layer = 0. Layer count = 6 if type is Cube, image layer count if type is 1D/2D/CubeArray, 1 otherwise.
		std::optional<vk::ImageAspectFlags> aspects; /// Default: All aspects.
		std::optional<vk::ComponentMapping> swizzle; /// Default: Identity.
		std::optional<bool> srgb_conversion; /// Default: false.
	};

	struct StorageDescriptorOverrides
	{
		std::optional<vk::ImageViewType> type; /// Default: 1D/2D/3D depending on image extent.
		std::optional<vk::Format> format; /// Default: Image format. Do not use an sRGB format here. sRGB is not supported for storage descriptors.
		std::optional<uint32_t> level; /// Default: 0.
		std::optional<Range<uint32_t>> layers; /// Default: First layer = 0. Layer count = 6 if type is Cube, image layer count if type is 1D/2D/CubeArray, 1 otherwise.
		std::optional<vk::ImageAspectFlags> aspects; /// Default: All aspects.
	};

	[[nodiscard]]
	static ImagePtr create(const DeviceSessionPtr& device_session, Desc&& desc);

	explicit Image(PrivateKey, const DeviceSessionPtr& device_session, Desc&& desc);

	Image(const Image&) = delete;
	Image(Image&&) = delete;

	Image& operator=(const Image&) = delete;
	Image& operator=(Image&&) = delete;

	~Image();

	[[nodiscard]]
	const DeviceSessionPtr& getDeviceSession() const;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::Image& getHandle();

	[[nodiscard]]
	uint32_t getSampledDescriptorHandle(const SampledDescriptorOverrides& overrides = {});

	[[nodiscard]]
	uint32_t getStorageDescriptorHandle(const StorageDescriptorOverrides& overrides = {});

private:
	struct SampledDescriptorInfo
	{
		vk::ImageViewType type{};
		vk::Format format{};
		Range<uint32_t> levels{};
		Range<uint32_t> layers{};
		vk::ImageAspectFlags aspects{};
		vk::ComponentMapping swizzle{};

		auto operator<=>(const SampledDescriptorInfo&) const = default;
	};

	struct StorageDescriptorInfo
	{
		vk::ImageViewType type{};
		vk::Format format{};
		uint32_t level{};
		Range<uint32_t> layers{};
		vk::ImageAspectFlags aspects{};

		auto operator<=>(const StorageDescriptorInfo&) const = default;
	};

	DeviceSessionPtr m_device_session;

	Desc m_desc;

	vk::Image m_handle{};
	vma::Allocation m_alloc{};

	vk::ImageViewType m_default_view_type{};
	vk::ImageAspectFlags m_default_view_aspects{};

	std::flat_map<SampledDescriptorInfo, uint32_t> m_sampled_cache;
	std::flat_map<StorageDescriptorInfo, uint32_t> m_storage_cache;

	void createImage();

	uint32_t calcDefaultLayerCount(vk::ImageViewType type);
};
}
