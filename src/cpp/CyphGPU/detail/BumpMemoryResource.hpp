#pragma once

#include <memory_resource>

namespace cgpu::detail
{
class BumpMemoryResource final : public std::pmr::monotonic_buffer_resource
{
public:
	using std::pmr::monotonic_buffer_resource::monotonic_buffer_resource;
};
}
