#pragma once

#include <CyphGPU/CommandContextSlot.hpp>
#include <CyphGPU/Queue.hpp>

#include <memory>
#include <stack>
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
	DeviceSessionPtr m_device_session;

	std::shared_ptr<CommandContextSlot> m_current_slot;
	std::stack<std::shared_ptr<CommandContextSlot>> m_available_slots;
	std::vector<std::shared_ptr<CommandContextSlot>> m_pending_slots;

	void beginSlot();
	void endSlot();

	void recycleFinishedSlots();
};
}
