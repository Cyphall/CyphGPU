#pragma once

#include <CyphGPU/DependencyObject.hpp>
#include <CyphGPU/fwd.hpp>

#include <boost/optional.hpp>
#include <vulkan/vulkan.hpp>

namespace cgpu
{
class Context final : public DependencyObject<Context>
{
	class PrivateKey
	{};

public:
	enum class Capability : uint8_t
	{
		eCore = 1 << 0,
		eSurfaceWin32 = 1 << 1,
		eSurfaceMetal = 1 << 2,
		eSurfaceXcb = 1 << 3,
		eSurfaceXlib = 1 << 4,
		eSurfaceWayland = 1 << 5,
	};

	using Capabilities = vk::Flags<Capability>;

	static constexpr uint32_t VULKAN_API_VERSION = vk::ApiVersion14;

	[[nodiscard]]
	static ContextRef create();

	explicit Context(PrivateKey);

	[[nodiscard]]
	const vk::detail::DispatchLoaderDynamic& getDispatcher() const;

	[[nodiscard]]
	Capabilities getCapabilities() const;

	[[nodiscard]]
	std::vector<std::string> getAvailableLayers() const;

private:
	friend class ContextSession;

	struct CapabilityData
	{
		std::vector<const char*> extensions;
	};

	vk::detail::DynamicLoader m_dynamic_loader{};
	vk::detail::DispatchLoaderDynamic m_dispatcher{};

	Capabilities m_capabilities{};

	[[nodiscard]]
	static boost::optional<const CapabilityData&> getCapabilityData(Capability capability);
};
}

// NOLINTBEGIN(readability-identifier-naming)
template<>
struct vk::FlagTraits<cgpu::Context::Capability>
{
	static VULKAN_HPP_CONST_OR_CONSTEXPR bool isBitmask = true;
	static VULKAN_HPP_CONST_OR_CONSTEXPR cgpu::Context::Capabilities allFlags =
		cgpu::Context::Capability::eCore |
		cgpu::Context::Capability::eSurfaceWin32 |
		cgpu::Context::Capability::eSurfaceMetal |
		cgpu::Context::Capability::eSurfaceXcb |
		cgpu::Context::Capability::eSurfaceXlib |
		cgpu::Context::Capability::eSurfaceWayland;
};

// NOLINTEND(readability-identifier-naming)
