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

	explicit PassContext(CommandRecorder& rec);

	static vk::AccessFlags2 toVk(StorageAccess access);
};
}
