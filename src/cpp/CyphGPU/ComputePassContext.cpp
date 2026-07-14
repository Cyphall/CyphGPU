#include "ComputePassContext.hpp"

#include <CyphGPU/CommandRecorder.hpp>

void cgpu::ComputePassContext::bindPipelineStates(const ComputeShaderStatePtr& compute_shader_state)
{
	if (compute_shader_state == m_current_compute_shader_state)
	{
		return;
	}

	m_rec->bindPipelineStates(
		compute_shader_state
	);

	m_current_compute_shader_state = compute_shader_state;
}

void cgpu::ComputePassContext::dispatch(const glm::uvec3& group_count)
{
	m_rec->dispatch(
		group_count
	);
}
