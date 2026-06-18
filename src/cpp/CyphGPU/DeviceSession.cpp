#include "DeviceSession.hpp"

#include <CyphGPU/Context.hpp>
#include <CyphGPU/Device.hpp>
#include <CyphGPU/Queue.hpp>
#include <CyphGPU/Utils.hpp>

#include <magic_enum/magic_enum.hpp>
#include <unordered_set>

namespace
{
struct QueueFamilyInfo
{
	uint32_t available_queue_count{};
	vk::QueueFlags caps{};
	uint32_t num_caps{};
};

std::vector<QueueFamilyInfo> queryAvailableQueues(const cgpu::DevicePtr& device)
{
	std::vector<vk::QueueFamilyProperties> queue_family_properties = device->getHandle().getQueueFamilyProperties(device->getContextSession()->getDispatcher());

	std::vector<QueueFamilyInfo> infos;
	infos.reserve(queue_family_properties.size());
	for (const vk::QueueFamilyProperties& properties : device->getHandle().getQueueFamilyProperties(device->getContextSession()->getDispatcher()))
	{
		QueueFamilyInfo& info = infos.emplace_back();
		info.available_queue_count = properties.queueCount;
		info.caps = properties.queueFlags;
		info.num_caps = static_cast<vk::QueueFlags::MaskType>(properties.queueFlags);
	}

	return infos;
}

std::optional<uint32_t> tryReserveBestQueue(std::span<QueueFamilyInfo> available_queues, vk::QueueFlags required_caps)
{
	std::optional<uint32_t> best_queue_family;
	for (uint32_t i = 0; i < available_queues.size(); i++)
	{
		// Not compatible
		if ((available_queues[i].caps & required_caps) != required_caps)
		{
			continue;
		}

		// No queue available for this family
		if (available_queues[i].available_queue_count == 0)
		{
			continue;
		}

		// Current candidate is better
		if (best_queue_family && available_queues[i].num_caps > available_queues[*best_queue_family].num_caps)
		{
			continue;
		}

		best_queue_family = i;
	}

	if (best_queue_family)
	{
		available_queues[*best_queue_family].available_queue_count--;
	}

	return best_queue_family;
}
}

cgpu::DeviceSessionPtr cgpu::DeviceSession::create(const DevicePtr& device, Desc&& desc)
{
	return std::make_shared<cgpu::DeviceSession>(PrivateKey{}, device, std::move(desc));
}

cgpu::DeviceSession::DeviceSession(PrivateKey, const DevicePtr& device, Desc&& desc):
	m_device{device},
	m_desc{std::move(desc)},
	m_dispatcher{device->getContextSession()->getDispatcher()}
{
	createDevice();
	createAllocator();
	createMemoryPools();
	createDescriptorHeaps();
}

cgpu::DeviceSession::~DeviceSession()
{
	m_vertex_input_state_cache.clear();

	m_allocator.destroyBuffer(m_resource_heap.buffer, m_resource_heap.alloc);
	m_allocator.destroyBuffer(m_sampler_heap.buffer, m_sampler_heap.alloc);

	for (const vma::Pool& pool : m_memory_pools)
	{
		m_allocator.destroy(pool);
	}

	m_allocator.destroy();

	m_async_transfer_queue.reset();
	m_async_compute_queue.reset();
	m_async_graphics_queue.reset();
	m_main_queue.reset();

	m_handle.destroy(nullptr, m_dispatcher);
}

const cgpu::DevicePtr& cgpu::DeviceSession::getDevice() const
{
	return m_device;
}

const cgpu::DeviceSession::Desc& cgpu::DeviceSession::getDesc() const
{
	return m_desc;
}

const vk::detail::DispatchLoaderDynamic& cgpu::DeviceSession::getDispatcher() const
{
	return m_dispatcher;
}

const vk::Device& cgpu::DeviceSession::getHandle()
{
	return m_handle;
}

cgpu::QueuePtr cgpu::DeviceSession::getMainQueue() const
{
	return {shared_from_this(), m_main_queue.get()};
}

cgpu::QueuePtr cgpu::DeviceSession::getAsyncGraphicsQueue() const
{
	return {shared_from_this(), m_async_graphics_queue.get()};
}

cgpu::QueuePtr cgpu::DeviceSession::getAsyncComputeQueue() const
{
	return {shared_from_this(), m_async_compute_queue.get()};
}

cgpu::QueuePtr cgpu::DeviceSession::getAsyncTransferQueue() const
{
	return {shared_from_this(), m_async_transfer_queue.get()};
}

std::pair<uint32_t, vk::HostAddressRangeEXT> cgpu::DeviceSession::Heap::reserveIndex()
{
	uint32_t index;
	{
		std::unique_lock lock{mutex};
		index = available_indices.back();
		available_indices.pop_back();
	}

	vk::HostAddressRangeEXT dst;
	dst.address = host_ptr + descriptor_size * index;
	dst.size = descriptor_size;

	return {index, dst};
}

void cgpu::DeviceSession::Heap::releaseIndex(uint32_t index)
{
	available_indices.emplace_back(index);
}

template<class T>
T& cgpu::DeviceSession::MetaObjectCache<T>::get(DeviceSession& device_session, T::Desc&& desc)
{
	std::unique_lock lock{m_mutex};
	auto [it, inserted] = m_map.try_emplace(desc);
	if (inserted)
	{
		it->second = std::make_unique<T>(typename T::PrivateKey{}, device_session, std::move(desc));
	}

	return *it->second;
}

template<class T>
void cgpu::DeviceSession::MetaObjectCache<T>::clear()
{
	m_map.clear();
}

void cgpu::DeviceSession::createDevice()
{
	// Reserve queues

	std::vector<QueueFamilyInfo> available_queues = queryAvailableQueues(getDevice());

	std::optional<uint32_t> main_queue_family = tryReserveBestQueue(
		available_queues,
		vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer
	);

	std::optional<uint32_t> async_graphics_queue_family = tryReserveBestQueue(
		available_queues,
		vk::QueueFlagBits::eGraphics
	);

	std::optional<uint32_t> async_compute_queue_family = tryReserveBestQueue(
		available_queues,
		vk::QueueFlagBits::eCompute
	);

	std::optional<uint32_t> async_transfer_queue_family = tryReserveBestQueue(
		available_queues,
		vk::QueueFlagBits::eTransfer
	);

	// Main queue is mandatory, other queues are optional
	if (!main_queue_family)
	{
		throw std::runtime_error("Could not find a suitable main queue.");
	}

	std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
	std::vector<std::vector<float>> queue_priorities;

	auto try_register_queue = [&](const std::optional<uint32_t>& family, float priority) -> std::optional<uint32_t> {
		if (!family)
		{
			return std::nullopt;
		}

		std::optional<size_t> index;
		for (size_t i = 0; i < queue_create_infos.size(); i++)
		{
			if (queue_create_infos[i].queueFamilyIndex == *family)
			{
				index = i;
				break;
			}
		}

		if (!index)
		{
			vk::DeviceQueueCreateInfo& info = queue_create_infos.emplace_back();
			info.flags = vk::DeviceQueueCreateFlagBits::eInternallySynchronizedKHR;
			info.queueFamilyIndex = *family;
			info.queueCount = *family;
			info.pQueuePriorities = nullptr;

			queue_priorities.emplace_back();

			index = queue_create_infos.size() - 1;
		}

		queue_priorities[*index].emplace_back(priority);

		queue_create_infos[*index].queueCount = static_cast<uint32_t>(queue_priorities[*index].size());
		queue_create_infos[*index].pQueuePriorities = queue_priorities[*index].data();

		return queue_create_infos[*index].queueCount - 1;
	};

	std::optional<uint32_t> main_queue_index = try_register_queue(main_queue_family, 1.0f);
	std::optional<uint32_t> async_graphics_queue_index = try_register_queue(async_graphics_queue_family, 0.0f);
	std::optional<uint32_t> async_compute_queue_index = try_register_queue(async_compute_queue_family, 0.0f);
	std::optional<uint32_t> async_transfer_queue_index = try_register_queue(async_transfer_queue_family, 0.0f);

	// Prepare features & extensions

	std::unordered_set<const char*, cgpu::StringHash, cgpu::StringEqualTo> unique_extensions;
	DynamicFeatureChain feature_chain;
	for (Device::Capability capability : magic_enum::enum_values<Device::Capability>())
	{
		if (!(m_device->getCapabilities() & capability))
		{
			continue;
		}

		auto capability_data = cgpu::Device::getCapabilityData(capability);
		unique_extensions.insert(capability_data->extensions.begin(), capability_data->extensions.end());
		capability_data->feature_callback(feature_chain);
	}

	std::vector<const char*> extensions{unique_extensions.begin(), unique_extensions.end()};

	// Create device

	vk::DeviceCreateInfo device_create_info;
	device_create_info.pNext = feature_chain.getHead();
	device_create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
	device_create_info.pQueueCreateInfos = queue_create_infos.data();
	device_create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	device_create_info.ppEnabledExtensionNames = extensions.data();

	m_handle = m_device->getHandle().createDevice(device_create_info, nullptr, m_dispatcher);

	m_dispatcher.init(m_handle);

	// Create queues

	auto try_create_queue = [&](const std::optional<uint32_t>& family, const std::optional<uint32_t>& index) -> std::shared_ptr<Queue> {
		if (!family || !index)
		{
			return {};
		}

		if (!std::ranges::contains(m_active_queue_families, *family))
		{
			m_active_queue_families.emplace_back(*family);
		}

		vk::DeviceQueueInfo2 info;
		info.flags = vk::DeviceQueueCreateFlagBits::eInternallySynchronizedKHR;
		info.queueFamilyIndex = *family;
		info.queueIndex = *index;

		vk::Queue queue = m_handle.getQueue2(info, m_dispatcher);

		return std::make_shared<Queue>(Queue::PrivateKey{}, *this, queue);
	};

	m_main_queue = try_create_queue(main_queue_family, main_queue_index);
	m_async_graphics_queue = try_create_queue(async_graphics_queue_family, async_graphics_queue_index);
	m_async_compute_queue = try_create_queue(async_compute_queue_family, async_compute_queue_index);
	m_async_transfer_queue = try_create_queue(async_transfer_queue_family, async_transfer_queue_index);

	// Some queue types may not have a dedicated queue.
	// In this case, use the main queue.
	if (!m_async_graphics_queue)
	{
		m_async_graphics_queue = m_main_queue;
	}
	if (!m_async_compute_queue)
	{
		m_async_compute_queue = m_main_queue;
	}
	if (!m_async_transfer_queue)
	{
		m_async_transfer_queue = m_main_queue;
	}
}

void cgpu::DeviceSession::createAllocator()
{
	vma::AllocatorCreateFlags flags;
	flags |= vma::AllocatorCreateFlagBits::eBufferDeviceAddress;
	flags |= vma::AllocatorCreateFlagBits::eKhrMaintenance4;
	flags |= vma::AllocatorCreateFlagBits::eKhrMaintenance5;
	if (m_device->getCapabilities() & Device::Capability::eMemoryBudget)
	{
		flags |= vma::AllocatorCreateFlagBits::eExtMemoryBudget;
	}
	if (m_device->getCapabilities() & Device::Capability::eMemoryPriority)
	{
		flags |= vma::AllocatorCreateFlagBits::eExtMemoryPriority;
	}

	vma::VulkanFunctions functions;
	functions.vkGetInstanceProcAddr = m_dispatcher.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr = m_dispatcher.vkGetDeviceProcAddr;

	vma::AllocatorCreateInfo info;
	info.flags = flags;
	info.physicalDevice = m_device->getHandle();
	info.device = m_handle;
	info.preferredLargeHeapBlockSize = 0;
	info.pAllocationCallbacks = nullptr;
	info.pDeviceMemoryCallbacks = nullptr;
	info.pHeapSizeLimit = nullptr;
	info.pVulkanFunctions = &functions;
	info.instance = m_device->getContextSession()->getHandle();
	info.vulkanApiVersion = Context::VULKAN_API_VERSION;
	info.pTypeExternalMemoryHandleTypes = nullptr;

	m_allocator = vma::createAllocator(info);
}

void cgpu::DeviceSession::createMemoryPools()
{
	auto create_pool = [&](vk::MemoryPropertyFlags flags, float priority) {
		vma::AllocationCreateInfo alloc_info;
		alloc_info.flags = {};
		alloc_info.usage = vma::MemoryUsage::eUnknown;
		alloc_info.requiredFlags = flags;
		alloc_info.preferredFlags = {};
		alloc_info.memoryTypeBits = std::numeric_limits<uint32_t>::max();
		alloc_info.pool = nullptr;
		alloc_info.pUserData = nullptr;
		alloc_info.priority = 0.5f;

		uint32_t memory_type_index = m_allocator.findMemoryTypeIndex(
			std::numeric_limits<uint32_t>::max(),
			alloc_info
		);

		vma::PoolCreateInfo pool_info;
		pool_info.memoryTypeIndex = memory_type_index;
		pool_info.flags = {};
		pool_info.blockSize = 0;
		pool_info.minBlockCount = 0;
		pool_info.maxBlockCount = 0;
		pool_info.priority = priority;
		pool_info.minAllocationAlignment = 0;
		pool_info.pMemoryAllocateNext = nullptr;

		return m_allocator.createPool(pool_info);
	};

	m_memory_pools[static_cast<size_t>(MemoryType::eGPULowPrio)] = create_pool(
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		0.5f
	);

	m_memory_pools[static_cast<size_t>(MemoryType::eGPUHighPrio)] = create_pool(
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		1.0f
	);

	m_memory_pools[static_cast<size_t>(MemoryType::eCPUCached)] = create_pool(
		vk::MemoryPropertyFlagBits::eHostVisible |
			vk::MemoryPropertyFlagBits::eHostCoherent |
			vk::MemoryPropertyFlagBits::eHostCached,
		0.0f
	);

	m_memory_pools[static_cast<size_t>(MemoryType::eCPUUncached)] = create_pool(
		vk::MemoryPropertyFlagBits::eHostVisible |
			vk::MemoryPropertyFlagBits::eHostCoherent,
		0.0f
	);

	m_memory_pools[static_cast<size_t>(MemoryType::eCPUVisibleGPU)] = create_pool(
		vk::MemoryPropertyFlagBits::eDeviceLocal |
			vk::MemoryPropertyFlagBits::eHostVisible |
			vk::MemoryPropertyFlagBits::eHostCoherent,
		0.0f
	);
}

void cgpu::DeviceSession::createDescriptorHeaps()
{
	auto create_heap = [&](Heap& heap, uint32_t count, vk::DeviceSize descriptor_size, vk::DeviceSize reserved_range) {
		vk::BufferCreateInfo buffer_info;
		buffer_info.flags = {};
		buffer_info.size = count * descriptor_size + reserved_range;
		buffer_info.usage = vk::BufferUsageFlagBits::eDescriptorHeapEXT | vk::BufferUsageFlagBits::eShaderDeviceAddress;
		buffer_info.sharingMode = vk::SharingMode::eExclusive;
		buffer_info.queueFamilyIndexCount = 0;
		buffer_info.pQueueFamilyIndices = nullptr;

		vma::AllocationCreateInfo alloc_create_info;
		alloc_create_info.flags = vma::AllocationCreateFlagBits::eMapped;
		alloc_create_info.usage = vma::MemoryUsage::eUnknown;
		alloc_create_info.requiredFlags = {};
		alloc_create_info.preferredFlags = {};
		alloc_create_info.memoryTypeBits = {};
		alloc_create_info.pool = getMemoryPool(MemoryType::eCPUVisibleGPU);
		alloc_create_info.pUserData = nullptr;
		alloc_create_info.priority = 0.0f;

		vma::AllocationInfo alloc_info;
		auto [alloc, buffer] = m_allocator.createBuffer(buffer_info, alloc_create_info, alloc_info);

		vk::BufferDeviceAddressInfo bda_info;
		bda_info.buffer = buffer;

		vk::DeviceAddress device_ptr = m_handle.getBufferAddress(bda_info, m_dispatcher);

		heap.buffer = buffer;
		heap.alloc = alloc;
		heap.host_ptr = static_cast<std::byte*>(alloc_info.pMappedData);
		heap.descriptor_size = descriptor_size;
		heap.bind_heap_info.heapRange.address = device_ptr;
		heap.bind_heap_info.heapRange.size = buffer_info.size;
		heap.bind_heap_info.reservedRangeOffset = count * descriptor_size;
		heap.bind_heap_info.reservedRangeSize = reserved_range;

		heap.available_indices.reserve(count - 1);
		for (uint32_t i = count - 1; i > 0; i--)
		{
			heap.available_indices.emplace_back(i);
		}
	};

	const auto& props = m_device->getProperties<vk::PhysicalDeviceDescriptorHeapPropertiesEXT>();

	// The specs guarantee at least this many descriptors
	create_heap(m_sampler_heap, 4000, props.samplerDescriptorSize, props.minSamplerHeapReservedRange);
	create_heap(m_resource_heap, 1015808, props.imageDescriptorSize, props.minResourceHeapReservedRange);
}

const vma::Allocator& cgpu::DeviceSession::getAllocator() const
{
	return m_allocator;
}

const vma::Pool& cgpu::DeviceSession::getMemoryPool(MemoryType type) const
{
	return m_memory_pools[static_cast<size_t>(type)];
}

std::span<const uint32_t> cgpu::DeviceSession::getActiveQueueFamilies() const
{
	return m_active_queue_families;
}

uint32_t cgpu::DeviceSession::createResourceDescriptor(const vk::ResourceDescriptorInfoEXT& info)
{
	auto [index, dst] = m_resource_heap.reserveIndex();

	m_handle.writeResourceDescriptorsEXT(info, dst, m_dispatcher);

	return index;
}

uint32_t cgpu::DeviceSession::createSamplerDescriptor(const vk::SamplerCreateInfo& info)
{
	auto [index, dst] = m_resource_heap.reserveIndex();

	m_handle.writeSamplerDescriptorsEXT(info, dst, m_dispatcher);

	return index;
}

void cgpu::DeviceSession::deleteResourceDescriptor(uint32_t index)
{
	m_resource_heap.releaseIndex(index);
}

const vk::BindHeapInfoEXT& cgpu::DeviceSession::getResourceBindHeapInfo() const
{
	return m_resource_heap.bind_heap_info;
}

const vk::BindHeapInfoEXT& cgpu::DeviceSession::getSamplerBindHeapInfo() const
{
	return m_sampler_heap.bind_heap_info;
}

cgpu::VertexInputState& cgpu::DeviceSession::getVertexInputState(VertexInputState::Desc&& desc)
{
	return m_vertex_input_state_cache.get(*this, std::move(desc));
}
