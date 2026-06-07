#pragma once

#include <CyphGPU/DependencyObject.hpp>
#include <CyphGPU/fwd.hpp>

#include <vulkan/vulkan.hpp>

namespace cgpu
{
class ContextSession final : public DependencyObject<ContextSession>
{
	class PrivateKey
	{};

public:
	struct Desc
	{
		// Required
		std::string application_name;

		// Optional
		uint32_t application_version{0};
		std::vector<const char*> enabled_layers{};
	};

	[[nodiscard]]
	static ContextSessionRef create(const ContextRef& context, Desc&& desc);

	explicit ContextSession(PrivateKey, const ContextRef& context, Desc&& desc);

	~ContextSession() override;

	[[nodiscard]]
	ContextRef getContext() const;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::detail::DispatchLoaderDynamic& getDispatcher() const;

	[[nodiscard]]
	const vk::Instance& getHandle() const;

	[[nodiscard]]
	std::vector<DeviceRef> getDevices() const;

private:
	DependencyParent<Context> m_context;

	Desc m_desc;

	vk::detail::DispatchLoaderDynamic m_dispatcher;

	vk::Instance m_handle{};
	vk::DebugUtilsMessengerEXT m_messenger{};

	std::vector<std::unique_ptr<Device>> m_devices;

	void createInstance();
	void createDebugMessenger();
	void queryDevices();
};
}
