#pragma once

#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/fwd.hpp>
#include <CyphGPU/Image.hpp>
#include <CyphGPU/PassContext.hpp>

#include <optional>

namespace cgpu
{
enum class GraphicsStage : uint8_t
{
	eVertex = 1 << 0,
	eGeometry = 1 << 1,
	eFragment = 1 << 2,
};

using GraphicsStages = vk::Flags<GraphicsStage>;

class GraphicsPassContext final : public PassContext
{
public:
	[[nodiscard]]
	SampledImageHandle getSampledImageDescriptor(const ImagePtr& image, GraphicsStages stages, const Image::SampledDescriptorOverrides& overrides = {});

	[[nodiscard]]
	StorageImageHandle getStorageImageDescriptor(const ImagePtr& image, GraphicsStages stages, StorageAccess access, const Image::StorageDescriptorOverrides& overrides = {});

	template<class T>
	[[nodiscard]]
	T* getBufferDevicePtr(const BufferPtr& buffer, GraphicsStages stages, StorageAccess access, vk::DeviceSize offset = 0)
	{
		registerStorageBufferIndirectAccess(buffer, stages, access);
		return buffer->getDevicePtrIndirect<T>(offset);
	}

	[[nodiscard]]
	UniformTexelBufferHandle getUniformTexelBufferDescriptor(const BufferPtr& buffer, GraphicsStages stages, vk::Format format, const Buffer::UniformTexelDescriptorOverrides& overrides = {});

	[[nodiscard]]
	StorageTexelBufferHandle getStorageTexelBufferDescriptor(const BufferPtr& buffer, GraphicsStages stages, StorageAccess access, vk::Format format, const Buffer::StorageTexelDescriptorOverrides& overrides = {});

	[[nodiscard]]
	vk::DeviceAddress getTLASDevicePtr(const TLASPtr& tlas, GraphicsStages stages);

	void registerSampledImageIndirectAccess(const ImagePtr& image, GraphicsStages stages);
	void registerStorageImageIndirectAccess(const ImagePtr& image, GraphicsStages stages, StorageAccess access);
	void registerSampledBufferIndirectAccess(const BufferPtr& buffer, GraphicsStages stages);
	void registerStorageBufferIndirectAccess(const BufferPtr& buffer, GraphicsStages stages, StorageAccess access);
	void registerTLASIndirectAccess(const TLASPtr& tlas, GraphicsStages stages);

	// ----- Commands -----

	void bindPipelineStates(
		const VertexInputStatePtr& vertex_input_state,
		const PreRasterizationShaderStatePtr& pre_rasterization_shader_state,
		const FragmentShaderStatePtr& fragment_shader_state,
		const FragmentOutputStatePtr& fragment_output_state
	);

	void bindIndexBuffer(
		const BufferPtr& buffer,
		vk::IndexType index_type,
		std::optional<Range<vk::DeviceSize>> range = {}
	);

	void draw(
		uint32_t vertex_count,
		uint32_t instance_count,
		uint32_t first_vertex,
		uint32_t first_instance,
		const void* data,
		size_t size,
		size_t alignment
	);

	template<class T>
	void draw(
		uint32_t vertex_count,
		uint32_t instance_count,
		uint32_t first_vertex,
		uint32_t first_instance,
		const T& data
	)
	{
		draw(
			vertex_count,
			instance_count,
			first_vertex,
			first_instance,
			&data,
			sizeof(T),
			alignof(T)
		);
	}

	void drawIndexed(
		uint32_t index_count,
		uint32_t instance_count,
		uint32_t first_index,
		int32_t vertex_offset,
		uint32_t first_instance,
		const void* data,
		size_t size,
		size_t alignment
	);

	template<class T>
	void drawIndexed(
		uint32_t index_count,
		uint32_t instance_count,
		uint32_t first_index,
		int32_t vertex_offset,
		uint32_t first_instance,
		const T& data
	)
	{
		drawIndexed(
			index_count,
			instance_count,
			first_index,
			vertex_offset,
			first_instance,
			&data,
			sizeof(T),
			alignof(T)
		);
	}

	void setViewport(
		const vk::Viewport& viewport
	);

	void setScissor(
		const vk::Rect2D& scissor
	);

private:
	friend class CommandRecorder;

	std::optional<VertexInputStatePtr> m_current_vertex_input_state;
	std::optional<PreRasterizationShaderStatePtr> m_current_pre_rasterization_shader_state;
	std::optional<FragmentShaderStatePtr> m_current_fragment_shader_state;
	std::optional<FragmentOutputStatePtr> m_current_fragment_output_state;

	using PassContext::PassContext;

	static vk::PipelineStageFlags2 toVk(GraphicsStages stages);
};
}

// NOLINTBEGIN(readability-identifier-naming)
template<>
struct vk::FlagTraits<cgpu::GraphicsStage>
{
	static VULKAN_HPP_CONST_OR_CONSTEXPR bool isBitmask = true;
	static VULKAN_HPP_CONST_OR_CONSTEXPR cgpu::GraphicsStages allFlags =
		cgpu::GraphicsStage::eVertex |
		cgpu::GraphicsStage::eGeometry |
		cgpu::GraphicsStage::eFragment;
};

// NOLINTEND(readability-identifier-naming)
