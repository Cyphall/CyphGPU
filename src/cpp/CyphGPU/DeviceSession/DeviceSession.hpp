#pragma once

#include <CyphGPU/fwd.hpp>

#include <vulkan/vulkan.hpp>

namespace cgpu
{
class DeviceSession final : public std::enable_shared_from_this<DeviceSession>
{
	class PrivateKey
	{};

public:
	struct Desc
	{
	};

	[[nodiscard]]
	static DeviceSessionRef create(const DeviceRef& device, Desc&& desc);

	explicit DeviceSession(PrivateKey, const DeviceRef& device, Desc&& desc);

	DeviceSession(const DeviceSession&) = delete;
	DeviceSession(DeviceSession&&) = delete;

	DeviceSession& operator=(const DeviceSession&) = delete;
	DeviceSession& operator=(DeviceSession&&) = delete;

	~DeviceSession();

	[[nodiscard]]
	const DeviceRef& getDevice() const;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::detail::DispatchLoaderDynamic& getDispatcher() const;

	[[nodiscard]]
	const vk::Device& getHandle() const;

private:
	DeviceRef m_device;

	Desc m_desc;

	vk::detail::DispatchLoaderDynamic m_dispatcher;

	vk::Device m_handle{};

	std::shared_ptr<Queue> m_main_queue;
	std::shared_ptr<Queue> m_async_graphics_queue;
	std::shared_ptr<Queue> m_async_compute_queue;
	std::shared_ptr<Queue> m_async_transfer_queue;

	void createDevice();
};
}
