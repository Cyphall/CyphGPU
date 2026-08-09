#pragma once

#include <CyphGPU/fwd.hpp>
#include <CyphGPU/PassContext.hpp>

#include <glm/glm.hpp>
#include <optional>

namespace cgpu
{
class CommandRecorder;

class ComputePassContext final : public PassContext
{
public:
	// ----- Commands -----

	void dispatch(
		const ComputeShaderStatePtr& compute_shader_state,
		const glm::uvec3& group_count,
		const void* data,
		size_t size,
		size_t alignment
	);

	template<class T>
	void dispatch(
		const ComputeShaderStatePtr& compute_shader_state,
		const glm::uvec3& group_count,
		const T& data
	)
	{
		dispatch(
			compute_shader_state,
			group_count,
			&data,
			sizeof(T),
			alignof(T)
		);
	}

private:
	friend class CommandRecorder;

	struct DispatchCmd
	{
		ComputeShaderStatePtr compute_shader_state;
		glm::uvec3 group_count;
		vk::DeviceAddress params_gpu_ptr;
	};

	detail::BumpList<DispatchCmd> m_dispatch_cmds;

	explicit ComputePassContext(CommandRecorder& rec, detail::BumpMemoryResource& bump_memory);

	[[nodiscard]]
	const detail::BumpList<DispatchCmd>& getDispatchCmds() const;
};
}
