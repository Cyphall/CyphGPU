#pragma once

#include <CyphGPU/CommandRecorder.hpp>
#include <CyphGPU/detail/BumpMemoryResource.hpp>
#include <CyphGPU/fwd.hpp>

#include <flat_map>
#include <unordered_map>

namespace cgpu
{
class CommandContext;

class CommandContextSlot : public std::enable_shared_from_this<CommandContextSlot>
{
	class PrivateKey
	{};

public:
	struct ParameterMemory
	{
		std::byte* cpu_ptr;
		vk::DeviceAddress gpu_ptr;
	};

	explicit CommandContextSlot(PrivateKey, const DeviceSessionPtr& device_session);

	~CommandContextSlot();

	CommandRecorder createRecorder(const QueuePtr& queue);

	[[nodiscard]]
	const DeviceSessionPtr& getDeviceSession() const;

	[[nodiscard]]
	ParameterMemory allocParameterMemory(vk::DeviceSize size, vk::DeviceSize alignment);

	[[nodiscard]]
	const std::flat_map<vk::Semaphore, uint64_t>& getFinishedSignals() const;

	void addFinishedSignal(const Queue::Signal& signal);

	void reset();

private:
	friend class CommandContext;
	friend class CommandRecorder;

	struct CommandPoolData
	{
		vk::CommandPool pool;
		std::vector<vk::CommandBuffer> available_cmdbufs;
		std::vector<vk::CommandBuffer> in_use_cmdbufs;
	};

	DeviceSessionPtr m_device_session;

	detail::BumpMemoryResource m_bump_memory{};

	// Indexed with queue family
	std::unordered_map<uint32_t, CommandPoolData> m_pools{};

	size_t m_num_cmdrec{0};
	bool m_high_cmdrecs_warning_emitted{false};

	vk::DeviceSize m_min_parambuf_alloc_alignment;
	std::vector<BufferPtr> m_free_parambufs{};
	std::vector<BufferPtr> m_used_parambufs{};
	vk::DeviceSize m_current_parambuf_offset{};

	std::flat_map<vk::Semaphore, uint64_t> m_finished_signals{};

	std::span<const BufferPtr> getParameterBuffers();

	vk::CommandBuffer getCommandBufferFromPool(CommandPoolData& pool_data);

	vk::CommandBuffer createImageInitCommandBuffer(const QueuePtr& queue);
};
}
