#include "Image.hpp"

#include <CyphGPU/DeviceSession/DeviceSession.hpp>

#include <cassert>
#include <flat_set>
#include <ranges>

cgpu::ImagePtr cgpu::Image::create(const DeviceSessionPtr& device_session, Desc&& desc)
{
	return std::make_shared<cgpu::Image>(PrivateKey{}, device_session, std::move(desc));
}

cgpu::Image::Image(PrivateKey, const DeviceSessionPtr& device_session, Desc&& desc):
	m_device_session{device_session},
	m_desc{std::move(desc)}
{
	createImage();
}

cgpu::Image::~Image()
{
	for (uint32_t index : m_storage_cache | std::views::values)
	{
		m_device_session->deleteResourceDescriptor(index);
	}
	for (uint32_t index : m_sampled_cache | std::views::values)
	{
		m_device_session->deleteResourceDescriptor(index);
	}

	m_device_session->getAllocator().destroyImage(m_handle, m_alloc);
}

const cgpu::DeviceSessionPtr& cgpu::Image::getDeviceSession() const
{
	return m_device_session;
}

const cgpu::Image::Desc& cgpu::Image::getDesc() const
{
	return m_desc;
}

const vk::Image& cgpu::Image::getHandle()
{
	return m_handle;
}

uint32_t cgpu::Image::getSampledDescriptorHandle(const SampledDescriptorOverrides& overrides)
{
	SampledDescriptorInfo info;
	info.type = overrides.type.value_or(m_default_view_type);
	info.format = overrides.format.value_or(m_desc.format);
	info.levels = overrides.levels.value_or(Range<uint32_t>{0, vk::RemainingMipLevels});
	info.layers = overrides.layers.value_or(Range<uint32_t>{0, calcDefaultLayerCount(info.type)});
	info.swizzle = overrides.swizzle.value_or(vk::ComponentMapping{});
	info.aspects = overrides.aspects.value_or(m_default_view_aspects);

	assert(getLinearEquivalent(info.format) == info.format);

	if (overrides.srgb_conversion.value_or(false))
	{
		info.format = getSrgbEquivalent(info.format);
	}

	auto [it, inserted] = m_sampled_cache.try_emplace(info);
	if (inserted)
	{
		vk::ImageViewCreateInfo view_info;
		view_info.flags = {};
		view_info.image = m_handle;
		view_info.viewType = info.type;
		view_info.format = info.format;
		view_info.components = info.swizzle;
		view_info.subresourceRange.aspectMask = info.aspects;
		view_info.subresourceRange.baseMipLevel = info.levels.offset;
		view_info.subresourceRange.levelCount = info.levels.size;
		view_info.subresourceRange.baseArrayLayer = info.layers.offset;
		view_info.subresourceRange.layerCount = info.layers.size;

		vk::ImageDescriptorInfoEXT typed_info;
		typed_info.pView = &view_info;
		typed_info.layout = vk::ImageLayout::eGeneral;

		vk::ResourceDescriptorInfoEXT desc_info;
		desc_info.type = vk::DescriptorType::eSampledImage;
		desc_info.data.pImage = &typed_info;

		it->second = m_device_session->createResourceDescriptor(desc_info);
	}

	return it->second;
}

uint32_t cgpu::Image::getStorageDescriptorHandle(const StorageDescriptorOverrides& overrides)
{
	StorageDescriptorInfo info;
	info.type = overrides.type.value_or(m_default_view_type);
	info.format = overrides.format.value_or(m_desc.format);
	info.level = overrides.level.value_or(0u);
	info.layers = overrides.layers.value_or(Range<uint32_t>{0, calcDefaultLayerCount(info.type)});
	info.swizzle = overrides.swizzle.value_or(vk::ComponentMapping{});
	info.aspects = overrides.aspects.value_or(m_default_view_aspects);

	assert(getLinearEquivalent(info.format) == info.format);

	auto [it, inserted] = m_storage_cache.try_emplace(info);
	if (inserted)
	{
		vk::ImageViewCreateInfo view_info;
		view_info.flags = {};
		view_info.image = m_handle;
		view_info.viewType = info.type;
		view_info.format = info.format;
		view_info.components = info.swizzle;
		view_info.subresourceRange.aspectMask = info.aspects;
		view_info.subresourceRange.baseMipLevel = info.level;
		view_info.subresourceRange.levelCount = 1;
		view_info.subresourceRange.baseArrayLayer = info.layers.offset;
		view_info.subresourceRange.layerCount = info.layers.size;

		vk::ImageDescriptorInfoEXT typed_info;
		typed_info.pView = &view_info;
		typed_info.layout = vk::ImageLayout::eGeneral;

		vk::ResourceDescriptorInfoEXT desc_info;
		desc_info.type = vk::DescriptorType::eStorageImage;
		desc_info.data.pImage = &typed_info;

		it->second = m_device_session->createResourceDescriptor(desc_info);
	}

	return it->second;
}

void cgpu::Image::createImage()
{
	assert(getLinearEquivalent(m_desc.format) == m_desc.format);

	vk::ImageType type;
	if (m_desc.extent.z > 1)
	{
		type = vk::ImageType::e3D;
		m_default_view_type = vk::ImageViewType::e3D;
	}
	else if (m_desc.extent.y > 1)
	{
		type = vk::ImageType::e2D;
		m_default_view_type = vk::ImageViewType::e2D;
	}
	else
	{
		type = vk::ImageType::e1D;
		m_default_view_type = vk::ImageViewType::e1D;
	}

	vk::ImageCreateFlags flags;
	flags |= vk::ImageCreateFlagBits::eMutableFormat;
	flags |= vk::ImageCreateFlagBits::eExtendedUsage;
	if (m_desc.allow_cube_view)
	{
		flags |= vk::ImageCreateFlagBits::eCubeCompatible;
	}
	if (m_desc.allow_2d_array_view)
	{
		flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;
	}
	if (m_desc.allow_block_texel_view)
	{
		flags |= vk::ImageCreateFlagBits::eBlockTexelViewCompatible;
	}

	std::flat_set<vk::Format> view_formats_set;
	view_formats_set.emplace(m_desc.format);
	view_formats_set.emplace(getSrgbEquivalent(m_desc.format));
	for (vk::Format format : m_desc.additional_view_formats)
	{
		view_formats_set.emplace(format);
		view_formats_set.emplace(getSrgbEquivalent(format));
	}

	std::vector<vk::Format> view_formats = std::move(view_formats_set).extract();

	vk::StructureChain<
		vk::ImageCreateInfo,
		vk::ImageFormatListCreateInfo>
		chain;

	auto& image_info = chain.get<vk::ImageCreateInfo>();
	image_info.flags = flags;
	image_info.imageType = type;
	image_info.format = m_desc.format;
	image_info.extent.width = m_desc.extent.x;
	image_info.extent.height = m_desc.extent.y;
	image_info.extent.depth = m_desc.extent.z;
	image_info.mipLevels = m_desc.levels;
	image_info.arrayLayers = m_desc.layers;
	image_info.samples = m_desc.samples;
	image_info.tiling = vk::ImageTiling::eOptimal;
	image_info.usage = m_desc.usages;
	image_info.sharingMode = vk::SharingMode::eExclusive;
	image_info.queueFamilyIndexCount = 0;
	image_info.pQueueFamilyIndices = nullptr;
	image_info.initialLayout = vk::ImageLayout::eUndefined;

	auto& image_format_list_info = chain.get<vk::ImageFormatListCreateInfo>();
	image_format_list_info.viewFormatCount = static_cast<uint32_t>(view_formats.size());
	image_format_list_info.pViewFormats = view_formats.data();

	vma::AllocationCreateInfo alloc_create_info;
	alloc_create_info.flags = {};
	alloc_create_info.usage = vma::MemoryUsage::eUnknown;
	alloc_create_info.requiredFlags = {};
	alloc_create_info.preferredFlags = {};
	alloc_create_info.memoryTypeBits = {};
	alloc_create_info.pool = m_device_session->getMemoryPool(m_desc.memory_type);
	alloc_create_info.pUserData = nullptr;
	alloc_create_info.priority = 0.0f;

	vma::AllocationInfo alloc_info;
	std::tie(m_alloc, m_handle) = m_device_session->getAllocator().createImage(image_info, alloc_create_info, alloc_info);

	m_device_session->getHandle().setDebugUtilsObjectNameEXT(m_handle, m_desc.name, m_device_session->getDispatcher());

	m_default_view_aspects = getAspects(m_desc.format);
}

uint32_t cgpu::Image::calcDefaultLayerCount(vk::ImageViewType type)
{
	if (type == vk::ImageViewType::eCube)
	{
		return 6;
	}
	if (type == vk::ImageViewType::e1DArray || type == vk::ImageViewType::e2DArray || type == vk::ImageViewType::eCubeArray)
	{
		return m_desc.layers;
	}
	return 1;
}
