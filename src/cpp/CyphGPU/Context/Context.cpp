#include "Context.hpp"

#include <CyphGPU/Utils.hpp>

#include <magic_enum/magic_enum.hpp>
#include <unordered_set>

cgpu::ContextRef cgpu::Context::create()
{
	cgpu::ContextRef ref = std::make_shared<cgpu::Context>(PrivateKey{});
	ref->m_weak_this = ref;
	return ref;
}

cgpu::Context::Context(PrivateKey)
{
	if (!m_dynamic_loader.success())
	{
		throw std::runtime_error("Failed to load Vulkan loader.");
	}

	m_dispatcher.init(m_dynamic_loader);

	uint32_t api_version = vk::enumerateInstanceVersion(m_dispatcher);
	if (api_version < VULKAN_API_VERSION)
	{
		throw std::runtime_error(
			std::format(
				"The Vulkan loader does not support Vulkan {}.{}.",
				vk::apiVersionMajor(VULKAN_API_VERSION),
				vk::apiVersionMinor(VULKAN_API_VERSION)
			)
		);
	}

	std::unordered_set<std::string, cgpu::StringHash, cgpu::StringEqualTo> supported_extensions;
	for (const vk::ExtensionProperties& extension_properties : vk::enumerateInstanceExtensionProperties(nullptr, m_dispatcher))
	{
		supported_extensions.emplace(extension_properties.extensionName.data());
	}

	auto is_capability_supported = [&](Capability capability) -> bool
	{
		boost::optional<const CapabilityData&> capability_data = getCapabilityData(capability);
		if (!capability_data)
		{
			return false;
		}

		for (const char* required_extension : capability_data->extensions)
		{
			if (!supported_extensions.contains(required_extension))
			{
				return false;
			}
		}

		return true;
	};

	for (Capability capability : magic_enum::enum_values<Capability>())
	{
		if (is_capability_supported(capability))
		{
			m_capabilities |= capability;
		}
	}
}

const vk::detail::DispatchLoaderDynamic& cgpu::Context::getDispatcher() const
{
	return m_dispatcher;
}

cgpu::Context::Capabilities cgpu::Context::getCapabilities() const
{
	return m_capabilities;
}

std::vector<std::string> cgpu::Context::getAvailableLayers() const
{
	std::vector<std::string> available_layers;
	for (const vk::LayerProperties& layer_properties : vk::enumerateInstanceLayerProperties(m_dispatcher))
	{
		available_layers.emplace_back(layer_properties.layerName.data());
	}
	return available_layers;
}

boost::optional<const cgpu::Context::CapabilityData&> cgpu::Context::getCapabilityData(Capability capability)
{
	static constexpr CapabilityData CORE{
		{
			vk::EXTDebugUtilsExtensionName,
		}
	};

#if defined(VK_USE_PLATFORM_WIN32_KHR)
	static constexpr CapabilityData SURFACE_WIN32{
		{
			vk::KHRSurfaceExtensionName,
			vk::KHRWin32SurfaceExtensionName,
		}
	};
#endif

#if defined(VK_USE_PLATFORM_METAL_EXT)
	static constexpr CapabilityData SURFACE_METAL{
		{
			vk::KHRSurfaceExtensionName,
			vk::EXTMetalSurfaceExtensionName,
		}
	};
#endif

#if defined(VK_USE_PLATFORM_XCB_KHR)
	static constexpr CapabilityData SURFACE_XCB{
		{
			vk::KHRSurfaceExtensionName,
			vk::KHRXcbSurfaceExtensionName,
		}
	};
#endif

#if defined(VK_USE_PLATFORM_XLIB_KHR)
	static constexpr CapabilityData SURFACE_XLIB{
		{
			vk::KHRSurfaceExtensionName,
			vk::KHRXlibSurfaceExtensionName,
		}
	};
#endif

#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
	static constexpr CapabilityData SURFACE_WAYLAND{
		{
			vk::KHRSurfaceExtensionName,
			vk::KHRWaylandSurfaceExtensionName,
		}
	};
#endif

	switch (capability)
	{
	case Capability::eCore: return CORE;
#if defined(VK_USE_PLATFORM_WIN32_KHR)
	case Capability::eSurfaceWin32: return SURFACE_WIN32;
#else
	case Capability::eSurfaceWin32: return boost::none;
#endif
#if defined(VK_USE_PLATFORM_METAL_EXT)
	case Capability::eSurfaceMetal: return SURFACE_METAL;
#else
	case Capability::eSurfaceMetal: return boost::none;
#endif
#if defined(VK_USE_PLATFORM_XCB_KHR)
	case Capability::eSurfaceXcb: return SURFACE_XCB;
#else
	case Capability::eSurfaceXcb: return boost::none;
#endif
#if defined(VK_USE_PLATFORM_XLIB_KHR)
	case Capability::eSurfaceXlib: return SURFACE_XLIB;
#else
	case Capability::eSurfaceXlib: return boost::none;
#endif
#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
	case Capability::eSurfaceWayland: return SURFACE_WAYLAND;
#else
	case Capability::eSurfaceWayland: return boost::none;
#endif
	default: std::terminate();
	}
}
