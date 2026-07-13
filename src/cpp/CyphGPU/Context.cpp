#include "Context.hpp"

#include <CyphGPU/Utils.hpp>

#include <magic_enum/magic_enum.hpp>
#include <unordered_set>
#include <utility>

cgpu::ContextPtr cgpu::Context::create(Desc&& desc)
{
	return std::make_shared<Context>(PrivateKey{}, std::move(desc));
}

cgpu::Context::Context(PrivateKey, Desc&& desc):
	m_desc{std::move(desc)}
{
	if (!m_dynamic_loader.success())
	{
		throw std::runtime_error("Failed to load Vulkan loader.");
	}

	m_dispatcher.init(m_dynamic_loader);

	checkCapabilitySupport();
}

const cgpu::Context::Desc& cgpu::Context::getDesc() const
{
	return m_desc;
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
	static CapabilityData core{
		{
			vk::EXTDebugUtilsExtensionName,
		}
	};

#if defined(VK_USE_PLATFORM_WIN32_KHR)
	static CapabilityData surface_win32{
		{
			vk::KHRSurfaceExtensionName,
			vk::KHRSurfaceMaintenance1ExtensionName,
			vk::KHRGetSurfaceCapabilities2ExtensionName,
			vk::EXTSwapchainColorSpaceExtensionName,
			vk::KHRWin32SurfaceExtensionName,
		}
	};
#endif

#if defined(VK_USE_PLATFORM_METAL_EXT)
	static CapabilityData surface_metal{
		{
			vk::KHRSurfaceExtensionName,
			vk::KHRSurfaceMaintenance1ExtensionName,
			vk::KHRGetSurfaceCapabilities2ExtensionName,
			vk::EXTSwapchainColorSpaceExtensionName,
			vk::EXTMetalSurfaceExtensionName,
		}
	};
#endif

#if defined(VK_USE_PLATFORM_XCB_KHR)
	static CapabilityData surface_xcb{
		{
			vk::KHRSurfaceExtensionName,
			vk::KHRSurfaceMaintenance1ExtensionName,
			vk::KHRGetSurfaceCapabilities2ExtensionName,
			vk::EXTSwapchainColorSpaceExtensionName,
			vk::KHRXcbSurfaceExtensionName,
		}
	};
#endif

#if defined(VK_USE_PLATFORM_XLIB_KHR)
	static CapabilityData surface_xlib{
		{
			vk::KHRSurfaceExtensionName,
			vk::KHRSurfaceMaintenance1ExtensionName,
			vk::KHRGetSurfaceCapabilities2ExtensionName,
			vk::EXTSwapchainColorSpaceExtensionName,
			vk::KHRXlibSurfaceExtensionName,
		}
	};
#endif

#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
	static CapabilityData surface_wayland{
		{
			vk::KHRSurfaceExtensionName,
			vk::KHRSurfaceMaintenance1ExtensionName,
			vk::KHRGetSurfaceCapabilities2ExtensionName,
			vk::EXTSwapchainColorSpaceExtensionName,
			vk::KHRWaylandSurfaceExtensionName,
		}
	};
#endif

	switch (capability)
	{
	case Capability::eCore: return core;
#if defined(VK_USE_PLATFORM_WIN32_KHR)
	case Capability::eSurfaceWin32: return surface_win32;
#else
	case Capability::eSurfaceWin32: return boost::none;
#endif
#if defined(VK_USE_PLATFORM_METAL_EXT)
	case Capability::eSurfaceMetal: return surface_metal;
#else
	case Capability::eSurfaceMetal: return boost::none;
#endif
#if defined(VK_USE_PLATFORM_XCB_KHR)
	case Capability::eSurfaceXcb: return surface_xcb;
#else
	case Capability::eSurfaceXcb: return boost::none;
#endif
#if defined(VK_USE_PLATFORM_XLIB_KHR)
	case Capability::eSurfaceXlib: return surface_xlib;
#else
	case Capability::eSurfaceXlib: return boost::none;
#endif
#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
	case Capability::eSurfaceWayland: return surface_wayland;
#else
	case Capability::eSurfaceWayland: return boost::none;
#endif
	default: std::unreachable();
	}
}

void cgpu::Context::checkCapabilitySupport()
{
	// If Vulkan version is not supported, nothing is considered supported
	uint32_t api_version = vk::enumerateInstanceVersion(m_dispatcher);
	if (api_version < VULKAN_API_VERSION)
	{
		return;
	}

	std::unordered_set<std::string, StringHash, StringEqualTo> supported_extensions;
	for (const vk::ExtensionProperties& extension_properties : vk::enumerateInstanceExtensionProperties(nullptr, m_dispatcher))
	{
		supported_extensions.emplace(extension_properties.extensionName.data());
	}

	auto is_capability_supported = [&](Capability capability) -> bool {
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
		// If core is not supported, nothing is considered supported
		else if (capability == Capability::eCore)
		{
			return;
		}
	}
}
