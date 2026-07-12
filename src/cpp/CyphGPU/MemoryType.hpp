#pragma once

namespace cgpu
{
enum class MemoryType : uint8_t
{
	eGPUHighPrio,
	eGPULowPrio,
	eCPUCached,
	eCPUUncached,
	eCPUVisibleGPU,
};
}
