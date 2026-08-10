#pragma once

#include <CyphGPU/detail/BumpMemoryResource.hpp>
#include <CyphGPU/fwd.hpp>
#include <CyphGPU/Queue.hpp>

#include <flat_map>
#include <memory>
#include <stack>
#include <unordered_map>
#include <vector>

namespace cgpu
{
class CommandContext
{
public:
	explicit CommandContext(const DeviceSessionPtr& device_session);

	CommandContext(const CommandContext&) = delete;
	CommandContext(CommandContext&&) = delete;

	CommandContext& operator=(const CommandContext&) = delete;
	CommandContext& operator=(CommandContext&&) = delete;

	CommandRecorder createRecorder(const QueuePtr& queue);

	/// Call this at the end of each frame/task/independent unit of work
	void finish();

private:
	friend class CommandRecorder;

	class Slot : public std::enable_shared_from_this<Slot>
	{
		class PrivateKey
		{};

	public:
		struct ParameterMemory
		{
			std::byte* cpu_ptr;
			vk::DeviceAddress gpu_ptr;
		};

		explicit Slot(PrivateKey, const DeviceSessionPtr& device_session);

		~Slot();

		CommandRecorder createRecorder(const QueuePtr& queue);

		[[nodiscard]]
		const DeviceSessionPtr& getDeviceSession() const;

		[[nodiscard]]
		ParameterMemory allocParameterMemory(vk::DeviceSize size, vk::DeviceSize alignment);

		[[nodiscard]]
		const std::flat_map<vk::Semaphore, uint64_t>& getFinishedSignals() const;

		void addFinishedSignal(const Queue::Signal& signal);

		void reset();

		vk::CommandBuffer createCommandBuffer(const QueuePtr& queue, vk::CommandBufferLevel level);

	private:
		friend class CommandContext;
		friend class CommandRecorder;

		struct CommandPoolData
		{
			vk::CommandPool pool;
			std::array<std::vector<vk::CommandBuffer>, 2> available_cmd_bufs;
			std::array<std::vector<vk::CommandBuffer>, 2> in_use_cmd_bufs;
		};

		DeviceSessionPtr m_device_session;

		detail::BumpMemoryResource m_bump_memory{};

		// Indexed with queue family
		std::unordered_map<uint32_t, CommandPoolData> m_pools{};

		size_t m_num_cmd_rec{0};
		bool m_high_cmd_recs_warning_emitted{false};

		vk::DeviceSize m_min_param_buf_alloc_alignment;
		std::vector<BufferPtr> m_free_param_bufs{};
		std::vector<BufferPtr> m_used_param_bufs{};
		vk::DeviceSize m_current_param_buf_offset{};

		std::flat_map<vk::Semaphore, uint64_t> m_finished_signals{};
	};

	DeviceSessionPtr m_device_session;

	std::shared_ptr<Slot> m_current_slot;
	std::stack<std::shared_ptr<Slot>> m_available_slots;
	std::vector<std::shared_ptr<Slot>> m_pending_slots;

	void beginSlot();
	void endSlot();

	void recycleFinishedSlots();
};
}
