#pragma once

#include <CyphGPU/CommandRecorder.hpp>
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

	DeviceSessionPtr m_device_session;

	// Indexed withy queue family
	std::unordered_map<uint32_t, vk::CommandPool> m_pools{};

	std::vector<BufferPtr> m_parameter_buffers{};
	uint64_t m_parameter_offset{};

	std::flat_map<vk::Semaphore, uint64_t> m_finished_signals;

	std::span<const BufferPtr> getParameterBuffers();
};
}
