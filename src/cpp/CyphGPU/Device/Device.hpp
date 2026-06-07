#pragma once

#include <CyphGPU/ContextSession/ContextSession.hpp>
#include <CyphGPU/DependencyObject.hpp>
#include <CyphGPU/Device/DynamicFeatureChain.hpp>
#include <CyphGPU/fwd.hpp>

#include <boost/optional.hpp>
#include <vulkan/vulkan.hpp>

namespace cgpu
{
class Device final : public DependencyObject<Device>
{
	class PrivateKey
	{};

public:
	enum class Capability : uint8_t
	{
		eCore = 1 << 0,
		eSwapchain = 1 << 1,
	};

	using Capabilities = vk::Flags<Capability>;

	explicit Device(PrivateKey, ContextSession& context_session, vk::PhysicalDevice physical_device);

	[[nodiscard]]
	const vk::PhysicalDevice& getHandle() const;

	[[nodiscard]]
	ContextSessionRef getContextSession() const;

	[[nodiscard]]
	Capabilities getCapabilities() const;

	template<class T>
	[[nodiscard]]
	const T& getProperties() const
	{
		auto [it, inserted] = m_properties_structs.try_emplace(T::structureType);
		if (inserted)
		{
			if constexpr (std::is_same_v<T, vk::PhysicalDeviceProperties2>)
			{
				it->second = std::make_shared<T>(
					m_physical_device.getProperties2(m_context_session->getDispatcher())
				);
			}
			else
			{
				it->second = std::make_shared<T>(
					m_physical_device.getProperties2<vk::PhysicalDeviceProperties2, T>(m_context_session->getDispatcher()).template get<T>()
				);
			}
		}

		return *static_cast<const T*>(it->second.get());
	}

private:
	friend class ContextSession;

	struct CapabilityData
	{
		std::vector<const char*> extensions;

		using FeatureCallback = void(DynamicFeatureChain& chain);
		FeatureCallback* feature_callback{};
	};

	DependencyParent<ContextSession> m_context_session;

	vk::PhysicalDevice m_physical_device;

	Capabilities m_capabilities{};

	mutable std::unordered_map<vk::StructureType, std::shared_ptr<void>> m_properties_structs{};

	[[nodiscard]]
	static boost::optional<const CapabilityData&> getCapabilityData(Capability capability);

	void checkCapabilitySupport();
};
}

// NOLINTBEGIN(readability-identifier-naming)
template<>
struct vk::FlagTraits<cgpu::Device::Capability>
{
	static VULKAN_HPP_CONST_OR_CONSTEXPR bool isBitmask = true;
	static VULKAN_HPP_CONST_OR_CONSTEXPR cgpu::Device::Capabilities allFlags =
		cgpu::Device::Capability::eCore |
		cgpu::Device::Capability::eSwapchain;
};

// NOLINTEND(readability-identifier-naming)
