#include "ComputePassContext.hpp"

#include <CyphGPU/CommandRecorder.hpp>

void cgpu::ComputePassContext::dispatch(
	const ComputeShaderStatePtr& compute_shader_state,
	const glm::uvec3& group_count,
	const void* data,
	size_t size,
	size_t alignment
)
{
	vk::DeviceAddress gpu_ptr = m_rec->writeParameters(
		data,
		size,
		alignment
	);

	m_dispatch_cmds.push_back({
		.compute_shader_state = compute_shader_state,
		.group_count = group_count,
		.params_gpu_ptr = gpu_ptr,
	});
}

cgpu::ComputePassContext::ComputePassContext(CommandRecorder& rec, detail::BumpMemoryResource& bump_memory):
	PassContext{rec},
	m_dispatch_cmds{detail::BumpAllocator{bump_memory}}
{
}

const cgpu::detail::BumpList<cgpu::ComputePassContext::DispatchCmd>& cgpu::ComputePassContext::getDispatchCmds() const
{
	return m_dispatch_cmds;
}
