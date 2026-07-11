#include "Buffer.hpp"

#include <CyphGPU/DeviceSession.hpp>

#include <ranges>

cgpu::BufferPtr cgpu::Buffer::create(const DeviceSessionPtr& device_session, Desc&& desc)
{
	return std::make_shared<cgpu::Buffer>(PrivateKey{}, device_session, std::move(desc));
}

cgpu::Buffer::Buffer(PrivateKey, const DeviceSessionPtr& device_session, Desc&& desc):
	m_device_session{device_session},
	m_desc{std::move(desc)}
{
	createBuffer();
}

cgpu::Buffer::~Buffer()
{
	for (uint32_t index : m_storage_texel_cache | std::views::values)
	{
		m_device_session->deleteResourceDescriptor(index);
	}
	for (uint32_t index : m_uniform_texel_cache | std::views::values)
	{
		m_device_session->deleteResourceDescriptor(index);
	}

	if (m_alloc)
	{
		vmaDestroyBuffer(m_device_session->getAllocator(), m_handle, *m_alloc);
	}
}

const cgpu::DeviceSessionPtr& cgpu::Buffer::getDeviceSession() const
{
	return m_device_session;
}

const cgpu::Buffer::Desc& cgpu::Buffer::getDesc() const
{
	return m_desc;
}

const vk::Buffer& cgpu::Buffer::getHandle()
{
	return m_handle;
}

vk::DeviceAddress cgpu::Buffer::getDevicePtr(vk::DeviceSize offset)
{
	return m_device_ptr + offset;
}

std::byte* cgpu::Buffer::getHostPtr(vk::DeviceSize offset)
{
	return m_host_ptr.value() + offset;
}

uint32_t cgpu::Buffer::getUniformTexelDescriptor(vk::Format format, const UniformTexelDescriptorOverrides& overrides)
{
	UniformTexelDescriptorInfo info;
	info.format = format;
	info.byte_range = overrides.byte_range.value_or(Range<vk::DeviceSize>{0, m_desc.size});

	auto [it, inserted] = m_uniform_texel_cache.try_emplace(info);
	if (inserted)
	{
		vk::TexelBufferDescriptorInfoEXT typed_info;
		typed_info.format = info.format;
		typed_info.addressRange.address = m_device_ptr + info.byte_range.offset;
		typed_info.addressRange.size = info.byte_range.size;

		vk::ResourceDescriptorInfoEXT desc_info;
		desc_info.type = vk::DescriptorType::eUniformTexelBuffer;
		desc_info.data.pTexelBuffer = &typed_info;

		it->second = m_device_session->createResourceDescriptor(desc_info);
	}

	return it->second;
}

uint32_t cgpu::Buffer::getStorageTexelDescriptor(vk::Format format, const StorageTexelDescriptorOverrides& overrides)
{
	StorageTexelDescriptorInfo info;
	info.format = format;
	info.byte_range = overrides.byte_range.value_or(Range<vk::DeviceSize>{0, m_desc.size});

	auto [it, inserted] = m_storage_texel_cache.try_emplace(info);
	if (inserted)
	{
		vk::TexelBufferDescriptorInfoEXT typed_info;
		typed_info.format = info.format;
		typed_info.addressRange.address = m_device_ptr + info.byte_range.offset;
		typed_info.addressRange.size = info.byte_range.size;

		vk::ResourceDescriptorInfoEXT desc_info;
		desc_info.type = vk::DescriptorType::eStorageTexelBuffer;
		desc_info.data.pTexelBuffer = &typed_info;

		it->second = m_device_session->createResourceDescriptor(desc_info);
	}

	return it->second;
}

void cgpu::Buffer::createBuffer()
{
	if (m_desc.existing_handle)
	{
		m_handle = m_desc.existing_handle->buffer;

		if (m_device_session->getMemoryPool(m_desc.memory_type).is_host_visible)
		{
			m_host_ptr = *m_desc.existing_handle->host_ptr;
		}
	}
	else
	{
		std::span<const uint32_t> queue_families = m_device_session->getActiveQueueFamilies();

		vk::StructureChain<
			vk::BufferCreateInfo,
			vk::BufferUsageFlags2CreateInfo>
			chain;

		auto& buffer_info = chain.get<vk::BufferCreateInfo>();
		buffer_info.flags = {};
		buffer_info.size = m_desc.size;
		buffer_info.usage = {};
		buffer_info.sharingMode = queue_families.size() > 1 ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive;
		buffer_info.queueFamilyIndexCount = static_cast<uint32_t>(queue_families.size());
		buffer_info.pQueueFamilyIndices = queue_families.data();

		auto& buffer_usage_info = chain.get<vk::BufferUsageFlags2CreateInfo>();
		buffer_usage_info.usage = m_desc.usages | vk::BufferUsageFlagBits2::eShaderDeviceAddress;

		VmaAllocationCreateInfo alloc_create_info{};
		alloc_create_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
		alloc_create_info.usage = VMA_MEMORY_USAGE_UNKNOWN;
		alloc_create_info.requiredFlags = {};
		alloc_create_info.preferredFlags = {};
		alloc_create_info.memoryTypeBits = {};
		alloc_create_info.pool = m_device_session->getMemoryPool(m_desc.memory_type).handle;
		alloc_create_info.pUserData = nullptr;
		alloc_create_info.priority = 0.0f;
		alloc_create_info.minAlignment = m_desc.min_alignment;

		VmaAllocationInfo alloc_info{};
		vk::detail::resultCheck(
			static_cast<vk::Result>(
				vmaCreateBuffer(
					m_device_session->getAllocator(),
					buffer_info,
					&alloc_create_info,
					reinterpret_cast<VkBuffer*>(&m_handle),
					&m_alloc.emplace(),
					&alloc_info
				)
			),
			"vmaCreateBuffer"
		);

		m_device_session->getHandle().setDebugUtilsObjectNameEXT(m_handle, m_desc.name, m_device_session->getDispatcher());

		if (alloc_info.pMappedData != nullptr)
		{
			m_host_ptr = static_cast<std::byte*>(alloc_info.pMappedData);
		}
	}

	vk::BufferDeviceAddressInfo bda_info;
	bda_info.buffer = m_handle;

	m_device_ptr = m_device_session->getHandle().getBufferAddress(bda_info, m_device_session->getDispatcher());
}
