#pragma once

#include <CyphGPU/fwd.hpp>
#include <CyphGPU/PassContext.hpp>

#include <optional>

namespace cgpu
{
class CommandRecorder;

class GraphicsPassContext final : public PassContext
{
public:
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
};
}
