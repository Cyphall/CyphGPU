#pragma once

#include <string_view>
#include <vulkan/vulkan.hpp>

namespace cgpu
{
struct StringHash : std::hash<std::string_view>
{
	using is_transparent = void;
};

struct StringLess : std::less<std::string_view>
{
	using is_transparent = void;
};

struct StringEqualTo : std::equal_to<std::string_view>
{
	using is_transparent = void;
};

template<class T>
struct Range
{
	T offset{};
	T size{};

	auto operator<=>(const Range&) const = default;
};

[[nodiscard]]
vk::Format getLinearEquivalent(vk::Format format);

[[nodiscard]]
vk::Format getSrgbEquivalent(vk::Format format);

[[nodiscard]]
vk::ImageAspectFlags getAspects(vk::Format format);

template<class T>
T alignUp(T size, T alignment)
{
	return ((size + alignment - 1) / alignment) * alignment;
}
}
