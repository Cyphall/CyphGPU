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

	void bindPipelineStates(
		const ComputeShaderStatePtr& compute_shader_state
	);

	void dispatch(
		const glm::uvec3& group_count
	);

private:
	friend class CommandRecorder;

	std::optional<ComputeShaderStatePtr> m_current_compute_shader_state;

	using PassContext::PassContext;
};
}
