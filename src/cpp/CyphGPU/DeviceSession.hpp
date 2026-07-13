#pragma once

#include <CyphGPU/ComputeShaderState.hpp>
#include <CyphGPU/FragmentOutputState.hpp>
#include <CyphGPU/FragmentShaderState.hpp>
#include <CyphGPU/fwd.hpp>
#include <CyphGPU/MemoryType.hpp>
#include <CyphGPU/PreRasterizationShaderState.hpp>
#include <CyphGPU/Sampler.hpp>
#include <CyphGPU/VertexInputState.hpp>

#include <magic_enum/magic_enum.hpp>
#include <mutex>
#include <unordered_map>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace cgpu
{
class CommandRecorder;

namespace detail
{
class ShaderChainBuilder;
}

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
	const vk::Device& getHandle();

	[[nodiscard]]
	QueuePtr getMainQueue() const;

	[[nodiscard]]
	QueuePtr getAsyncGraphicsQueue() const;

	[[nodiscard]]
	QueuePtr getAsyncComputeQueue() const;

	[[nodiscard]]
	QueuePtr getAsyncTransferQueue() const;

	void waitIdle();

private:
	friend class Buffer;
	friend class Image;
	friend class Sampler;
	friend class Swapchain;
	friend class VertexInputState;
	friend class PreRasterizationShaderState;
	friend class FragmentShaderState;
	friend class FragmentOutputState;
	friend class ComputeShaderState;
	friend class CommandRecorder;
	friend class detail::ShaderChainBuilder;

	struct MemoryPool
	{
		VmaPool handle{};
		bool is_host_visible{};
	};

	struct Heap
	{
		vk::Buffer buffer{};
		VmaAllocation alloc{};
		std::byte* host_ptr{};
		vk::DeviceSize descriptor_size{};
		vk::BindHeapInfoEXT bind_heap_info{};
		std::vector<uint32_t> available_indices{};
		std::mutex mutex{};

		[[nodiscard]]
		std::pair<uint32_t, vk::HostAddressRangeEXT> reserveIndex();

		void releaseIndex(uint32_t index);
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

	struct GraphicsPipelineKey
	{
		VertexInputState* vertex_input_state{};
		PreRasterizationShaderState* pre_rasterization_shader_state{};
		FragmentShaderState* fragment_shader_state{};
		FragmentOutputState* fragment_output_state{};

		bool operator==(const GraphicsPipelineKey&) const = default;
	};

	struct GraphicsPipelineKeyHasher
	{
		std::size_t operator()(const GraphicsPipelineKey& key) const noexcept;
	};

	struct GraphicsPipelineValue
	{
		vk::Pipeline fast_link_pipeline;
	};

	DevicePtr m_device;

	Desc m_desc;

	vk::detail::DispatchLoaderDynamic m_dispatcher;

	vk::Device m_handle{};

	std::vector<uint32_t> m_active_queue_families{};

	std::shared_ptr<Queue> m_main_queue{};
	std::shared_ptr<Queue> m_async_graphics_queue{};
	std::shared_ptr<Queue> m_async_compute_queue{};
	std::shared_ptr<Queue> m_async_transfer_queue{};

	VmaAllocator m_allocator{};

	std::array<MemoryPool, magic_enum::enum_count<MemoryType>()> m_memory_pools{};

	Heap m_sampler_heap{};
	Heap m_resource_heap{};
	std::vector<vk::DescriptorSetAndBindingMappingEXT> m_mappings{};

	MetaObjectCache<Sampler> m_sampler_cache{};
	MetaObjectCache<VertexInputState> m_vertex_input_state_cache{};
	MetaObjectCache<PreRasterizationShaderState> m_pre_rasterization_shader_state_cache{};
	MetaObjectCache<FragmentShaderState> m_fragment_shader_state_cache{};
	MetaObjectCache<FragmentOutputState> m_fragment_output_state_cache{};
	MetaObjectCache<ComputeShaderState> m_compute_shader_state_cache{};

	std::unordered_map<
		GraphicsPipelineKey,
		std::unique_ptr<GraphicsPipelineValue>,
		GraphicsPipelineKeyHasher>
		m_graphics_pipeline_cache{};
	std::mutex m_graphics_pipeline_cache_mutex{};

	void createDevice();
	void createAllocator();
	void createMemoryPools();
	void createDescriptorHeaps();

	[[nodiscard]]
	const VmaAllocator& getAllocator() const;

	[[nodiscard]]
	const MemoryPool& getMemoryPool(MemoryType type) const;

	std::span<const uint32_t> getActiveQueueFamilies() const;

	[[nodiscard]]
	uint32_t createResourceDescriptor(const vk::ResourceDescriptorInfoEXT& info);

	[[nodiscard]]
	uint32_t createSamplerDescriptor(const vk::SamplerCreateInfo& info);

	void deleteResourceDescriptor(uint32_t index);

	void deleteSamplerDescriptor(uint32_t index);

	[[nodiscard]]
	const vk::BindHeapInfoEXT& getResourceBindHeapInfo() const;

	[[nodiscard]]
	const vk::BindHeapInfoEXT& getSamplerBindHeapInfo() const;

	[[nodiscard]]
	Sampler& getSampler(Sampler::Desc&& desc);

	[[nodiscard]]
	std::span<const vk::DescriptorSetAndBindingMappingEXT> getMappings() const;

	[[nodiscard]]
	VertexInputState& getVertexInputState(VertexInputState::Desc&& desc);

	[[nodiscard]]
	PreRasterizationShaderState& getPreRasterizationShaderState(PreRasterizationShaderState::Desc&& desc);

	[[nodiscard]]
	FragmentShaderState& getFragmentShaderState(FragmentShaderState::Desc&& desc);

	[[nodiscard]]
	FragmentOutputState& getFragmentOutputState(FragmentOutputState::Desc&& desc);

	[[nodiscard]]
	ComputeShaderState& getComputeShaderState(ComputeShaderState::Desc&& desc);

	[[nodiscard]]
	vk::Pipeline linkGraphicsPipeline(
		const VertexInputStatePtr& vertex_input_state,
		const PreRasterizationShaderStatePtr& pre_rasterization_shader_state,
		const FragmentShaderStatePtr& fragment_shader_state,
		const FragmentOutputStatePtr& fragment_output_state,
		bool lto
	);

	[[nodiscard]]
	vk::Pipeline getGraphicsPipeline(
		const VertexInputStatePtr& vertex_input_state,
		const PreRasterizationShaderStatePtr& pre_rasterization_shader_state,
		const FragmentShaderStatePtr& fragment_shader_state,
		const FragmentOutputStatePtr& fragment_output_state
	);
};
}
