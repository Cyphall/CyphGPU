#pragma once

#include <CyphGPU/CommandRecorder.hpp>

#include <vulkan/vulkan.hpp>

namespace cgpu
{
enum class StorageAccess : uint8_t
{
	eReadonly,
	eWriteonly,
	eReadWrite,
};

class PassContext
{
protected:
	CommandRecorder* m_rec;
	CommandRecorder::CmdBase* m_cmd;

	explicit PassContext(CommandRecorder& rec, CommandRecorder::CmdBase& cmd);

	static vk::AccessFlags2 toVk(StorageAccess access);
};
}
