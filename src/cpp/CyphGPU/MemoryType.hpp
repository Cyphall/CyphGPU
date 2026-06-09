#pragma once

namespace cgpu
{
enum class MemoryType
{
	eGPUHighPrio,
	eGPULowPrio,
	eCPUCached,
	eCPUUncached,
	eCPUVisibleGPU,
};
}
