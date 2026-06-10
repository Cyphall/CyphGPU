#pragma once

#include <CyphGPU/fwd.hpp>
#include <CyphGPU/MemoryType.hpp>
#include <CyphGPU/Pipeline/VertexInputState.hpp>

#include <magic_enum/magic_enum.hpp>
#include <mutex>
#include <unordered_map>
#include <vulkan-memory-allocator-hpp/vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>

namespace cgpu
{
class DeviceSession final : public std::enable_shared_from_this<DeviceSession>
{
	class PrivateKey
	{};

public:
	struct Desc
	{
	};

	[[nodiscard]]
	static DeviceSessionPtr create(const DevicePtr& device, Desc&& desc);

	explicit DeviceSession(PrivateKey, const DevicePtr& device, Desc&& desc);

	DeviceSession(const DeviceSession&) = delete;
	DeviceSession(DeviceSession&&) = delete;

	DeviceSession& operator=(const DeviceSession&) = delete;
	DeviceSession& operator=(DeviceSession&&) = delete;

	~DeviceSession();

	[[nodiscard]]
	const DevicePtr& getDevice() const;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::detail::DispatchLoaderDynamic& getDispatcher() const;

	[[nodiscard]]
	const vk::Device& getHandle() const;

	[[nodiscard]]
	QueuePtr getMainQueue() const;

	[[nodiscard]]
	QueuePtr getAsyncGraphicsQueue() const;

	[[nodiscard]]
	QueuePtr getAsyncComputeQueue() const;

	[[nodiscard]]
	QueuePtr getAsyncTransferQueue() const;

private:
	friend class VertexInputState;

	struct Heap
	{
		vk::Buffer buffer{};
		vma::Allocation alloc{};
		vk::DeviceAddress device_ptr{};
		void* host_ptr{};
		vk::DeviceSize descriptor_size{};
		vk::DeviceSize reserved_range_offset{};
		std::vector<uint32_t> available_indices{};
	};

	template<class T>
	class MetaObjectCache
	{
	public:
		[[nodiscard]]
		T& get(DeviceSession& device_session, T::Desc&& desc);

		void clear();

	private:
		std::unordered_map<typename T::Desc, std::unique_ptr<T>> m_map{};
		std::mutex m_mutex{};
	};

	DevicePtr m_device;

	Desc m_desc;

	vk::detail::DispatchLoaderDynamic m_dispatcher;

	vk::Device m_handle{};

	std::shared_ptr<Queue> m_main_queue{};
	std::shared_ptr<Queue> m_async_graphics_queue{};
	std::shared_ptr<Queue> m_async_compute_queue{};
	std::shared_ptr<Queue> m_async_transfer_queue{};

	vma::Allocator m_allocator{};

	std::array<vma::Pool, magic_enum::enum_count<MemoryType>()> m_memory_pools{};

	Heap m_sampler_heap;
	Heap m_resource_heap;

	MetaObjectCache<VertexInputState> m_vertex_input_state_cache{};

	void createDevice();
	void createAllocator();
	void createMemoryPools();
	void createDescriptorHeaps();

	[[nodiscard]]
	const vma::Allocator& getAllocator() const;

	[[nodiscard]]
	const vma::Pool& getMemoryPool(MemoryType type) const;

	[[nodiscard]]
	VertexInputState& getVertexInputState(VertexInputState::Desc&& desc);
};
}
