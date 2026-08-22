#include "PassContext.hpp"

cgpu::PassContext::PassContext(CommandRecorder& rec, CommandRecorder::CmdBase& cmd):
	m_rec(&rec),
	m_cmd{&cmd}
{
}

vk::AccessFlags2 cgpu::PassContext::toVk(StorageAccess access)
{
	switch (access)
	{
	case StorageAccess::eReadonly: return vk::AccessFlagBits2::eShaderStorageRead;
	case StorageAccess::eWriteonly: return vk::AccessFlagBits2::eShaderStorageWrite;
	case StorageAccess::eReadWrite: return vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite;
	}

	std::unreachable();
}
