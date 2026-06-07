#include "ContextSession.hpp"

#include <CyphGPU/Context/Context.hpp>
#include <CyphGPU/Device/Device.hpp>
#include <CyphGPU/Utils.hpp>

#include <magic_enum/magic_enum.hpp>
#include <spdlog/spdlog.h>
#include <unordered_set>

namespace
{
VKAPI_ATTR vk::Bool32 VKAPI_CALL messageCallback(
	vk::DebugUtilsMessageSeverityFlagBitsEXT message_severity,
	vk::DebugUtilsMessageTypeFlagsEXT,
	const vk::DebugUtilsMessengerCallbackDataEXT* message_data,
	void*
)
{
	switch (message_severity)
	{
	case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
		spdlog::error(message_data->pMessage);
		break;
	case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
		spdlog::warn(message_data->pMessage);
		break;
	case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
		spdlog::info(message_data->pMessage);
		break;
	case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
		spdlog::debug(message_data->pMessage);
		break;
	}

	return VK_FALSE;
}
}

cgpu::ContextSessionRef cgpu::ContextSession::create(const ContextRef& context, Desc&& desc)
{
	cgpu::ContextSessionRef ref = std::make_shared<cgpu::ContextSession>(PrivateKey{}, context, std::move(desc));
	ref->m_weak_this = ref;
	return ref;
}

cgpu::ContextSession::ContextSession(PrivateKey, const ContextRef& context, Desc&& desc):
	m_context{context},
	m_desc{std::move(desc)},
	m_dispatcher{context->getDispatcher()}
{
	createInstance();
	createDebugMessenger();
	queryDevices();
}

cgpu::ContextSession::~ContextSession()
{
	m_instance.destroy(m_messenger, nullptr, m_dispatcher);
	m_instance.destroy(nullptr, m_dispatcher);
}

cgpu::ContextRef cgpu::ContextSession::getContext() const
{
	return m_context.get();
}

const cgpu::ContextSession::Desc& cgpu::ContextSession::getDesc() const
{
	return m_desc;
}

const vk::detail::DispatchLoaderDynamic& cgpu::ContextSession::getDispatcher() const
{
	return m_dispatcher;
}

const vk::Instance& cgpu::ContextSession::getHandle() const
{
	return m_instance;
}

std::vector<cgpu::DeviceRef> cgpu::ContextSession::getDevices() const
{
	std::vector<cgpu::DeviceRef> devices;
	devices.reserve(m_devices.size());
	for (const auto& device : m_devices)
	{
		devices.emplace_back(device->asRef());
	}

	return devices;
}

void cgpu::ContextSession::createInstance()
{
	vk::ApplicationInfo app_info;
	app_info.pApplicationName = m_desc.application_name.c_str();
	app_info.applicationVersion = m_desc.application_version;
	app_info.pEngineName = "CyphGPU";
	app_info.engineVersion = 0;
	app_info.apiVersion = Context::VULKAN_API_VERSION;

	std::unordered_set<const char*, cgpu::StringHash, cgpu::StringEqualTo> unique_extensions;
	for (Context::Capability capability : magic_enum::enum_values<Context::Capability>())
	{
		if (!(m_context->getCapabilities() & capability))
		{
			continue;
		}

		auto capability_data = cgpu::Context::getCapabilityData(capability);
		unique_extensions.insert(capability_data->extensions.begin(), capability_data->extensions.end());
	}

	std::vector<const char*> extensions{unique_extensions.begin(), unique_extensions.end()};

	vk::InstanceCreateInfo create_info;
	create_info.flags = {};
	create_info.pApplicationInfo = &app_info;
	create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	create_info.ppEnabledExtensionNames = extensions.data();
	create_info.enabledLayerCount = static_cast<uint32_t>(m_desc.enabled_layers.size());
	create_info.ppEnabledLayerNames = m_desc.enabled_layers.data();

	m_instance = vk::createInstance(create_info, nullptr, m_dispatcher);

	m_dispatcher.init(m_instance);
}

void cgpu::ContextSession::createDebugMessenger()
{
	vk::DebugUtilsMessengerCreateInfoEXT create_info;
	create_info.flags = {};
	create_info.messageSeverity =
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
	create_info.messageType =
		vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
		vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
		vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
	create_info.pfnUserCallback = &messageCallback;
	create_info.pUserData = nullptr;

	m_messenger = m_instance.createDebugUtilsMessengerEXT(create_info, nullptr, m_dispatcher);
}

void cgpu::ContextSession::queryDevices()
{
	for (auto physical_device : m_instance.enumeratePhysicalDevices(m_dispatcher))
	{
		m_devices.emplace_back(std::make_unique<Device>(Device::PrivateKey{}, *this, physical_device));
	}
}
