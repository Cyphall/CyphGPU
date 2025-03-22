#include "ContextSession.hpp"

#include <CyphGPU/Utils.hpp>

#include <magic_enum/magic_enum.hpp>
#include <spdlog/spdlog.h>
#include <unordered_set>

namespace
{
VKAPI_ATTR vk::Bool32 VKAPI_CALL messageCallback(
	vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	vk::DebugUtilsMessageTypeFlagsEXT,
	const vk::DebugUtilsMessengerCallbackDataEXT* messageData,
	void*
)
{
	switch (messageSeverity)
	{
	case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
		spdlog::error(messageData->pMessage);
		break;
	case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
		spdlog::warn(messageData->pMessage);
		break;
	case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
		spdlog::info(messageData->pMessage);
		break;
	case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
		spdlog::debug(messageData->pMessage);
		break;
	}

	return false;
}
}

cgpu::ContextSessionRef cgpu::ContextSession::create(const ContextRef& context, Desc&& desc)
{
	cgpu::ContextSessionRef ref = std::make_shared<cgpu::ContextSession>(PrivateKey{}, context, std::move(desc));
	ref->m_weakThis = ref;
	return ref;
}

cgpu::ContextSession::ContextSession(PrivateKey, const ContextRef& context, Desc&& desc):
	DependencyObjectChild{context},
	m_desc{std::move(desc)},
	m_dispatcher{m_parent.getDispatcher()}
{
	// Create instance
	{
		vk::ApplicationInfo appInfo;
		appInfo.pApplicationName = m_desc.applicationName.c_str();
		appInfo.applicationVersion = m_desc.applicationVersion;
		appInfo.pEngineName = "CyphGPU";
		appInfo.engineVersion = 0;
		appInfo.apiVersion = Context::VULKAN_API_VERSION;

		std::unordered_set<const char*, cgpu::StringHash, cgpu::StringEqualTo> uniqueExtensions;
		for (Context::Capability capability : magic_enum::enum_values<Context::Capability>())
		{
			if (!(m_parent.getCapabilities() & capability))
			{
				continue;
			}

			const Context::CapabilityData& capabilityData = m_parent.getCapabilityData(capability).value();
			uniqueExtensions.insert(capabilityData.extensions.begin(), capabilityData.extensions.end());
		}

		std::vector<const char*> extensions{uniqueExtensions.begin(), uniqueExtensions.end()};

		vk::InstanceCreateInfo createInfo;
		createInfo.flags = {};
		createInfo.pApplicationInfo = &appInfo;
		createInfo.enabledExtensionCount = extensions.size();
		createInfo.ppEnabledExtensionNames = extensions.data();
		createInfo.enabledLayerCount = m_desc.enabledLayers.size();
		createInfo.ppEnabledLayerNames = m_desc.enabledLayers.data();

		m_instance = vk::createInstanceUnique(createInfo, nullptr, m_dispatcher);
	}

	// Query instance functions
	{
		m_dispatcher.init(*m_instance);
	}

	// Create debug messenger
	{
		vk::DebugUtilsMessengerCreateInfoEXT createInfo;
		createInfo.flags = {};
		createInfo.messageSeverity =
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
		createInfo.messageType =
			vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
			vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
			vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
		createInfo.pfnUserCallback = &messageCallback;
		createInfo.pUserData = nullptr;

		m_messenger = m_instance->createDebugUtilsMessengerEXTUnique(createInfo, nullptr, m_dispatcher);
	}
}

const cgpu::ContextSession::Desc& cgpu::ContextSession::getDesc() const
{
	return m_desc;
}

const vk::detail::DispatchLoaderDynamic& cgpu::ContextSession::getDispatcher() const
{
	return m_dispatcher;
}