#pragma once

#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/CommandRecorder.hpp>
#include <CyphGPU/Image.hpp>
#include <CyphGPU/ShaderTypes.hpp>

#include <glm/glm.hpp>

namespace cgpu
{
class CommandRecorder;

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
