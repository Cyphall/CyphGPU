#include "GraphicsPassContext.hpp"

#include <CyphGPU/CommandRecorder.hpp>

void cgpu::GraphicsPassContext::bindPipelineStates(
	const VertexInputStatePtr& vertex_input_state,
	const PreRasterizationShaderStatePtr& pre_rasterization_shader_state,
	const FragmentShaderStatePtr& fragment_shader_state,
	const FragmentOutputStatePtr& fragment_output_state
)
{
	if (vertex_input_state == m_current_vertex_input_state &&
	    pre_rasterization_shader_state == m_current_pre_rasterization_shader_state &&
	    fragment_shader_state == m_current_fragment_shader_state &&
	    fragment_output_state == m_current_fragment_output_state)
	{
		return;
	}

	m_rec->bindPipelineStates(
		vertex_input_state,
		pre_rasterization_shader_state,
		fragment_shader_state,
		fragment_output_state
	);

	m_current_vertex_input_state = vertex_input_state;
	m_current_pre_rasterization_shader_state = pre_rasterization_shader_state;
	m_current_fragment_shader_state = fragment_shader_state;
	m_current_fragment_output_state = fragment_output_state;
}

void cgpu::GraphicsPassContext::pushParameters(uint32_t slot, const void* data, size_t size, size_t alignment)
{
	m_rec->pushParameters(
		slot,
		data,
		size,
		alignment
	);
}

void cgpu::GraphicsPassContext::draw(
	uint32_t vertex_count,
	uint32_t instance_count,
	uint32_t first_vertex,
	uint32_t first_instance
)
{
	m_rec->draw(
		vertex_count,
		instance_count,
		first_vertex,
		first_instance
	);
}

void cgpu::GraphicsPassContext::drawIndexed(
	uint32_t index_count,
	uint32_t instance_count,
	uint32_t first_index,
	int32_t vertex_offset,
	uint32_t first_instance
)
{
	m_rec->drawIndexed(
		index_count,
		instance_count,
		first_index,
		vertex_offset,
		first_instance
	);
}

void cgpu::GraphicsPassContext::setViewport(
	const vk::Viewport& viewport
)
{
	m_rec->setViewport(viewport);
}

void cgpu::GraphicsPassContext::setScissor(
	const vk::Rect2D& scissor
)
{
	m_rec->setScissor(scissor);
}

cgpu::GraphicsPassContext::GraphicsPassContext(CommandRecorder& rec):
	m_rec(&rec)
{
}
