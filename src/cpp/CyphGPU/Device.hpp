#pragma once

#include <CyphGPU/ContextSession.hpp>
#include <CyphGPU/fwd.hpp>

#include <boost/optional.hpp>
#include <mutex>
#include <shared_mutex>
#include <vulkan/vulkan.hpp>

namespace cgpu
{
namespace detail
{
class DynamicFeatureChain;
}

class Device final
{
	class PrivateKey
	{};

public:
	enum class Capability : uint8_t
	{
		eCore = 1 << 0,
		eSwapchain = 1 << 1,
		eMemoryBudget = 1 << 2,
		eMemoryPriority = 1 << 3,
		ePageableDeviceLocalMemory = 1 << 4,
		eUnifiedImageLayouts = 1 << 5,
		eRayTracing = 1 << 6,
	};

	using Capabilities = vk::Flags<Capability>;

	explicit Device(PrivateKey, ContextSession& context_session, vk::PhysicalDevice physical_device);

	Device(const Device&) = delete;
	Device(Device&&) = delete;

	Device& operator=(const Device&) = delete;
	Device& operator=(Device&&) = delete;

	[[nodiscard]]
	ContextSessionPtr getContextSession() const;

	[[nodiscard]]
	const vk::PhysicalDevice& getHandle();

	[[nodiscard]]
	Capabilities getCapabilities() const;

	[[nodiscard]]
	uint32_t getVulkanVersion() const;

	[[nodiscard]]
	uint32_t getDriverVersion() const;

	[[nodiscard]]
	uint32_t getVendorID() const;

	[[nodiscard]]
	uint32_t getDeviceID() const;

	[[nodiscard]]
	vk::PhysicalDeviceType getType() const;

	[[nodiscard]]
	const std::string& getDeviceName() const;

	[[nodiscard]]
	vk::DriverId getDriverID() const;

	[[nodiscard]]
	const std::string& getDriverName() const;

	[[nodiscard]]
	const std::string& getDriverInfo() const;

	template<class T>
	[[nodiscard]]
	const T& getProperties() const
	{
		std::unique_lock lock{m_properties_mutex};

		auto [it, inserted] = m_properties_structs.try_emplace(T::structureType);
		if (inserted)
		{
			if constexpr (std::is_same_v<T, vk::PhysicalDeviceProperties2>)
			{
				it->second = std::make_shared<T>(
					m_handle.getProperties2(m_context_session->getDispatcher())
				);
			}
			else
			{
				it->second = std::make_shared<T>(
					m_handle.getProperties2<vk::PhysicalDeviceProperties2, T>(m_context_session->getDispatcher()).template get<T>()
				);
			}
		}

		return *static_cast<const T*>(it->second.get());
	}

	[[nodiscard]]
	std::optional<vk::SurfaceFormatKHR> selectBestSurfaceFormat(
		const SurfacePtr& surface,
		std::span<const vk::SurfaceFormatKHR> formats
	) const;

	[[nodiscard]]
	std::optional<vk::CompositeAlphaFlagBitsKHR> selectBestAlphaMode(
		const SurfacePtr& surface,
		std::span<const vk::CompositeAlphaFlagBitsKHR> alpha_modes
	) const;

private:
	friend class ContextSession;
	friend class DeviceSession;

	struct CapabilityData
	{
		std::vector<const char*> extensions;

		using FeatureCallback = void(detail::DynamicFeatureChain& chain);
		FeatureCallback* feature_callback{};
	};

	ContextSession* m_context_session;

	vk::PhysicalDevice m_handle;

	Capabilities m_capabilities{};

	uint32_t m_vulkan_version{};
	uint32_t m_driver_version{};
	uint32_t m_vendor_id{};
	uint32_t m_device_id{};
	vk::PhysicalDeviceType m_type{};
	std::string m_device_name{};
	vk::DriverId m_driver_id{};
	std::string m_driver_name{};
	std::string m_driver_info{};

	mutable std::unordered_map<vk::StructureType, std::shared_ptr<void>> m_properties_structs{};
	mutable std::shared_mutex m_properties_mutex{};

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
		cgpu::Device::Capability::eSwapchain |
		cgpu::Device::Capability::eMemoryBudget |
		cgpu::Device::Capability::eMemoryPriority |
		cgpu::Device::Capability::ePageableDeviceLocalMemory |
		cgpu::Device::Capability::eUnifiedImageLayouts |
		cgpu::Device::Capability::eRayTracing;
};

// NOLINTEND(readability-identifier-naming)
