#pragma once

#include <glm/glm.hpp>
#include <optional>
#include <string_view>
#include <vulkan/vulkan.hpp>

#if __has_cpp_attribute(clang::require_explicit_initialization)
#	define CGPU_REQUIRED [[clang::require_explicit_initialization]]
#else
#	define CGPU_REQUIRED
#endif

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
constexpr T alignUp(T size, std::type_identity_t<T> alignment)
{
	return ((size + alignment - T{1}) / alignment) * alignment;
}

glm::uvec3 calcImageLevelExtent(const glm::uvec3& base_extent, uint32_t level);

glm::uvec3 calcImageExtentInBlocks(vk::Format format, const glm::uvec3& pixel_extent);

vk::DeviceSize calcImageByteSize(vk::Format format, const glm::uvec3& extent, uint32_t layers, std::optional<vk::ImageAspectFlags> aspects = std::nullopt);

vk::DeviceSize calcImageByteSize(vk::Format format, const glm::uvec3& base_extent, Range<uint32_t> levels, uint32_t layers, std::optional<vk::ImageAspectFlags> aspects = std::nullopt);

[[nodiscard]]
uint32_t calcImageMaxLevelCount(const glm::uvec3& extent);
}
