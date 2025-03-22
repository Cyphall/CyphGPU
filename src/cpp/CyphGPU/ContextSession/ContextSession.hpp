#pragma once

#include <CyphGPU/Context/Context.hpp>
#include <CyphGPU/DependencyObject.hpp>
#include <CyphGPU/fwd.hpp>

#include <vulkan/vulkan.hpp>

namespace cgpu
{
class ContextSession final : public DependencyObjectChild<ContextSession, Context>
{
	class PrivateKey
	{};

public:
	struct Desc
	{
		// Required
		std::string applicationName;

		// Optional
		uint32_t applicationVersion{0};
		std::vector<const char*> enabledLayers{};
	};

	[[nodiscard]]
	static ContextSessionRef create(const ContextRef& context, Desc&& desc);

	explicit ContextSession(PrivateKey, const ContextRef& context, Desc&& desc);

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::detail::DispatchLoaderDynamic& getDispatcher() const;

private:
	Desc m_desc;

	vk::detail::DispatchLoaderDynamic m_dispatcher;

	vk::UniqueInstance m_instance{};
	vk::UniqueDebugUtilsMessengerEXT m_messenger{};
};
}