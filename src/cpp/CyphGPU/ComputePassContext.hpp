#pragma once

#include <CyphGPU/fwd.hpp>

#include <cstdint>
#include <glm/glm.hpp>
#include <optional>

namespace cgpu
{
class CommandRecorder;

class ComputePassContext
{
public:

	// ----- Commands -----

	void bindPipelineStates(
		const ComputeShaderStatePtr& compute_shader_state
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

	void dispatch(
		const glm::uvec3& group_count
	);

private:
	friend class CommandRecorder;

	CommandRecorder* m_rec;

	std::optional<ComputeShaderStatePtr> m_current_compute_shader_state;

	explicit ComputePassContext(CommandRecorder& rec);
};
}
