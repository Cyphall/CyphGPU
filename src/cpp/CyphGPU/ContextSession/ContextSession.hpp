#pragma once

#include <CyphGPU/fwd.hpp>

#include <vulkan/vulkan.hpp>

namespace cgpu
{
class ContextSession final : public std::enable_shared_from_this<ContextSession>
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
	static ContextSessionPtr create(const ContextPtr& context, Desc&& desc);

	explicit ContextSession(PrivateKey, const ContextPtr& context, Desc&& desc);

	ContextSession(const ContextSession&) = delete;
	ContextSession(ContextSession&&) = delete;

	ContextSession& operator=(const ContextSession&) = delete;
	ContextSession& operator=(ContextSession&&) = delete;

	~ContextSession();

	[[nodiscard]]
	const ContextPtr& getContext() const;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::detail::DispatchLoaderDynamic& getDispatcher() const;

	[[nodiscard]]
	const vk::Instance& getHandle() const;

	[[nodiscard]]
	std::vector<DevicePtr> getDevices() const;

private:
	ContextPtr m_context;

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
