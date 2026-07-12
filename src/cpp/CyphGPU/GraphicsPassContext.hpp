#pragma once

#include <CyphGPU/fwd.hpp>

#include <optional>

namespace cgpu
{
class CommandRecorder;

class GraphicsPassContext
{
public:
	// ----- Commands -----

	void bindPipelineStates(
		const VertexInputStatePtr& vertex_input_state,
		const PreRasterizationShaderStatePtr& pre_rasterization_shader_state,
		const FragmentShaderStatePtr& fragment_shader_state,
		const FragmentOutputStatePtr& fragment_output_state
	);

	void pushParameters(
		uint32_t slot,
		const void* data,
		size_t size,
		size_t alignment
	);

	template<class T>
	void pushParameters(
		uint32_t slot,
		const T& data
	)
	{
		pushParameters(slot, &data, sizeof(T), alignof(T));
	}

	void draw(
		uint32_t vertex_count,
		uint32_t instance_count,
		uint32_t first_vertex,
		uint32_t first_instance
	);

	void drawIndexed(
		uint32_t index_count,
		uint32_t instance_count,
		uint32_t first_index,
		int32_t vertex_offset,
		uint32_t first_instance
		);

	void setViewport(
		const vk::Viewport& viewport
	);

	void setScissor(
		const vk::Rect2D& scissor
	);

private:
	friend class CommandRecorder;

	CommandRecorder* m_rec;

	std::optional<VertexInputStatePtr> m_current_vertex_input_state;
	std::optional<PreRasterizationShaderStatePtr> m_current_pre_rasterization_shader_state;
	std::optional<FragmentShaderStatePtr> m_current_fragment_shader_state;
	std::optional<FragmentOutputStatePtr> m_current_fragment_output_state;

	explicit GraphicsPassContext(CommandRecorder& rec);
};
}
