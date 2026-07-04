#include "DeviceSession.hpp"

#include <CyphGPU/Context.hpp>
#include <CyphGPU/Device.hpp>
#include <CyphGPU/HashExt.hpp>
#include <CyphGPU/Queue.hpp>
#include <CyphGPU/Utils.hpp>

#include <magic_enum/magic_enum.hpp>
#include <ranges>
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
	for (const vk::QueueFamilyProperties& properties : queue_family_properties)
	{
		QueueFamilyInfo& info = infos.emplace_back();
		info.available_queue_count = properties.queueCount;
		info.caps = properties.queueFlags;
		info.num_caps = std::popcount(static_cast<vk::QueueFlags::MaskType>(properties.queueFlags));
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
	if (!(m_device->getCapabilities() & Device::Capability::eCore))
	{
		throw std::logic_error("Cannot create a session for a device that does not support the Core capability.");
	}

	createDevice();
	createAllocator();
	createMemoryPools();
	createDescriptorHeaps();
}

cgpu::DeviceSession::~DeviceSession()
{
	for (const auto& graphics_pipeline : m_graphics_pipeline_cache | std::views::values)
	{
		m_handle.destroyPipeline(graphics_pipeline->fast_link_pipeline, nullptr, m_dispatcher);
	}

	m_fragment_output_state_cache.clear();
	m_fragment_shader_state_cache.clear();
	m_pre_rasterization_shader_state_cache.clear();
	m_vertex_input_state_cache.clear();
	m_sampler_cache.clear();

	vmaDestroyBuffer(m_allocator, m_resource_heap.buffer, m_resource_heap.alloc);
	vmaDestroyBuffer(m_allocator, m_sampler_heap.buffer, m_sampler_heap.alloc);

	for (const auto& pool : m_memory_pools)
	{
		vmaDestroyPool(m_allocator, pool.handle);
	}

	vmaDestroyAllocator(m_allocator);

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

void cgpu::DeviceSession::waitIdle()
{
	m_main_queue->waitIdle();
	m_async_graphics_queue->waitIdle();
	m_async_compute_queue->waitIdle();
	m_async_transfer_queue->waitIdle();
}

std::pair<uint32_t, vk::HostAddressRangeEXT> cgpu::DeviceSession::Heap::reserveIndex()
{
	uint32_t index{};
	{
		std::unique_lock lock{mutex};
		if (available_indices.empty())
		{
			throw std::runtime_error("Out of descriptor slots!");
		}
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
	std::unique_lock lock{mutex};
	available_indices.emplace_back(index);
}

std::size_t cgpu::DeviceSession::GraphicsPipelineKeyHasher::operator()(const GraphicsPipelineKey& key) const noexcept
{
	size_t seed = 0;
	cgpu::hashCombine(seed, key.vertex_input_state);
	cgpu::hashCombine(seed, key.pre_rasterization_shader_state);
	cgpu::hashCombine(seed, key.fragment_shader_state);
	cgpu::hashCombine(seed, key.fragment_output_state);
	return seed;
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
		vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute
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
	device_create_info.pNext = &feature_chain.get<vk::PhysicalDeviceFeatures2>();
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
	VmaAllocatorCreateFlags flags{};
	flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT;
	flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT;
	if (m_device->getCapabilities() & Device::Capability::eMemoryBudget)
	{
		flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}
	if (m_device->getCapabilities() & Device::Capability::eMemoryPriority)
	{
		flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
	}

	VmaVulkanFunctions functions{};
	functions.vkGetInstanceProcAddr = m_dispatcher.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr = m_dispatcher.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info{};
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

	vk::detail::resultCheck(
		static_cast<vk::Result>(vmaCreateAllocator(
			&info,
			&m_allocator
		)),
		"vmaCreateAllocator"
	);
}

void cgpu::DeviceSession::createMemoryPools()
{
	auto create_pool = [&](vk::MemoryPropertyFlags flags, float priority) -> MemoryPool {
		VmaAllocationCreateInfo alloc_info{};
		alloc_info.flags = {};
		alloc_info.usage = VMA_MEMORY_USAGE_UNKNOWN;
		alloc_info.requiredFlags = static_cast<VkMemoryPropertyFlags>(flags);
		alloc_info.preferredFlags = {};
		alloc_info.memoryTypeBits = std::numeric_limits<uint32_t>::max();
		alloc_info.pool = nullptr;
		alloc_info.pUserData = nullptr;
		alloc_info.priority = 0.5f;
		alloc_info.minAlignment = 0;

		uint32_t memory_type_index{};
		vk::detail::resultCheck(
			static_cast<vk::Result>(vmaFindMemoryTypeIndex(
				m_allocator,
				std::numeric_limits<uint32_t>::max(),
				&alloc_info,
				&memory_type_index
			)),
			"vmaFindMemoryTypeIndex"
		);

		VmaPoolCreateInfo pool_info{};
		pool_info.memoryTypeIndex = memory_type_index;
		pool_info.flags = {};
		pool_info.blockSize = 0;
		pool_info.minBlockCount = 0;
		pool_info.maxBlockCount = 0;
		pool_info.priority = priority;
		pool_info.minAllocationAlignment = 0;
		pool_info.pMemoryAllocateNext = nullptr;

		VmaPool pool{};
		vk::detail::resultCheck(
			static_cast<vk::Result>(
				vmaCreatePool(
					m_allocator,
					&pool_info,
					&pool
				)
			),
			"vmaCreatePool"
		);

		return {
			.handle = pool,
			.is_host_visible = static_cast<bool>(flags & vk::MemoryPropertyFlagBits::eHostVisible),
		};
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
		buffer_info.sharingMode = m_active_queue_families.size() > 1 ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive;
		buffer_info.queueFamilyIndexCount = static_cast<uint32_t>(m_active_queue_families.size());
		buffer_info.pQueueFamilyIndices = m_active_queue_families.data();

		VmaAllocationCreateInfo alloc_create_info{};
		alloc_create_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
		alloc_create_info.usage = VMA_MEMORY_USAGE_UNKNOWN;
		alloc_create_info.requiredFlags = {};
		alloc_create_info.preferredFlags = {};
		alloc_create_info.memoryTypeBits = {};
		alloc_create_info.pool = getMemoryPool(MemoryType::eCPUVisibleGPU).handle;
		alloc_create_info.pUserData = nullptr;
		alloc_create_info.priority = 0.0f;

		vk::Buffer buffer{};
		VmaAllocation alloc{};
		VmaAllocationInfo alloc_info{};
		vk::detail::resultCheck(
			static_cast<vk::Result>(
				vmaCreateBuffer(
					m_allocator,
					buffer_info,
					&alloc_create_info,
					reinterpret_cast<VkBuffer*>(&buffer),
					&alloc,
					&alloc_info
				)
			),
			"vmaCreateBuffer"
		);

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

	for (uint32_t i = 0; i < 4; i++)
	{
		auto& mapping = m_mappings.emplace_back();
		mapping.descriptorSet = 0;
		mapping.firstBinding = i;
		mapping.bindingCount = 1;
		mapping.resourceMask = vk::SpirvResourceTypeFlagBitsEXT::eUniformBuffer;
		mapping.source = vk::DescriptorMappingSourceEXT::ePushAddress;
		mapping.sourceData.pushAddressOffset = i * sizeof(vk::DeviceAddress);
	}
}

const VmaAllocator& cgpu::DeviceSession::getAllocator() const
{
	return m_allocator;
}

const cgpu::DeviceSession::MemoryPool& cgpu::DeviceSession::getMemoryPool(MemoryType type) const
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
	auto [index, dst] = m_sampler_heap.reserveIndex();

	m_handle.writeSamplerDescriptorsEXT(info, dst, m_dispatcher);

	return index;
}

void cgpu::DeviceSession::deleteResourceDescriptor(uint32_t index)
{
	m_resource_heap.releaseIndex(index);
}

void cgpu::DeviceSession::deleteSamplerDescriptor(uint32_t index)
{
	m_sampler_heap.releaseIndex(index);
}

const vk::BindHeapInfoEXT& cgpu::DeviceSession::getResourceBindHeapInfo() const
{
	return m_resource_heap.bind_heap_info;
}

const vk::BindHeapInfoEXT& cgpu::DeviceSession::getSamplerBindHeapInfo() const
{
	return m_sampler_heap.bind_heap_info;
}

cgpu::Sampler& cgpu::DeviceSession::getSampler(Sampler::Desc&& desc)
{
	return m_sampler_cache.get(*this, std::move(desc));
}

std::span<const vk::DescriptorSetAndBindingMappingEXT> cgpu::DeviceSession::getMappings() const
{
	return m_mappings;
}

cgpu::VertexInputState& cgpu::DeviceSession::getVertexInputState(VertexInputState::Desc&& desc)
{
	return m_vertex_input_state_cache.get(*this, std::move(desc));
}

cgpu::PreRasterizationShaderState& cgpu::DeviceSession::getPreRasterizationShaderState(PreRasterizationShaderState::Desc&& desc)
{
	return m_pre_rasterization_shader_state_cache.get(*this, std::move(desc));
}

cgpu::FragmentShaderState& cgpu::DeviceSession::getFragmentShaderState(FragmentShaderState::Desc&& desc)
{
	return m_fragment_shader_state_cache.get(*this, std::move(desc));
}

cgpu::FragmentOutputState& cgpu::DeviceSession::getFragmentOutputState(FragmentOutputState::Desc&& desc)
{
	return m_fragment_output_state_cache.get(*this, std::move(desc));
}

vk::Pipeline cgpu::DeviceSession::linkGraphicsPipeline(
	const VertexInputStatePtr& vertex_input_state,
	const PreRasterizationShaderStatePtr& pre_rasterization_shader_state,
	const FragmentShaderStatePtr& fragment_shader_state,
	const FragmentOutputStatePtr& fragment_output_state,
	bool lto
)
{
	std::array libraries = {
		vertex_input_state->getHandle(),
		pre_rasterization_shader_state->getHandle(),
		fragment_shader_state->getHandle(),
		fragment_output_state->getHandle(),
	};

	vk::StructureChain<
		vk::GraphicsPipelineCreateInfo,
		vk::PipelineCreateFlags2CreateInfo,
		vk::PipelineLibraryCreateInfoKHR>
		chain;

	auto& flags_create_info = chain.get<vk::PipelineCreateFlags2CreateInfo>();
	flags_create_info.flags = vk::PipelineCreateFlagBits2::eDescriptorHeapEXT;
	if (lto)
	{
		flags_create_info.flags |= vk::PipelineCreateFlagBits2::eLinkTimeOptimizationEXT;
	}

	auto& library_info = chain.get<vk::PipelineLibraryCreateInfoKHR>();
	library_info.libraryCount = static_cast<uint32_t>(libraries.size());
	library_info.pLibraries = libraries.data();

	return m_handle.createGraphicsPipeline(nullptr, chain.get(), nullptr, m_dispatcher).value;
}

vk::Pipeline cgpu::DeviceSession::getGraphicsPipeline(
	const VertexInputStatePtr& vertex_input_state,
	const PreRasterizationShaderStatePtr& pre_rasterization_shader_state,
	const FragmentShaderStatePtr& fragment_shader_state,
	const FragmentOutputStatePtr& fragment_output_state
)
{
	std::unique_lock lock{m_graphics_pipeline_cache_mutex};

	auto [it, inserted] = m_graphics_pipeline_cache.try_emplace({
		vertex_input_state.get(),
		pre_rasterization_shader_state.get(),
		fragment_shader_state.get(),
		fragment_output_state.get(),
	});
	if (inserted)
	{
		it->second = std::make_unique<GraphicsPipelineValue>();
		it->second->fast_link_pipeline = linkGraphicsPipeline(
			vertex_input_state,
			pre_rasterization_shader_state,
			fragment_shader_state,
			fragment_output_state,
			false
		);
	}

	return it->second->fast_link_pipeline;
}
