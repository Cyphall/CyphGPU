#pragma once

#include <CyphGPU/fwd.hpp>

#include <vulkan/vulkan.hpp>

namespace cgpu
{
class VertexInputState final
{
	class PrivateKey
	{};

public:
	struct Desc
	{
		// Optional
		vk::PrimitiveTopology topology{vk::PrimitiveTopology::eTriangleList};

		bool operator==(const Desc& other) const = default;
	};

	[[nodiscard]]
	static VertexInputStatePtr create(const DeviceSessionPtr& device_session, Desc&& desc);

	explicit VertexInputState(PrivateKey, DeviceSession& device_session, Desc&& desc);

	VertexInputState(const VertexInputState&) = delete;
	VertexInputState(VertexInputState&&) = delete;

	VertexInputState& operator=(const VertexInputState&) = delete;
	VertexInputState& operator=(VertexInputState&&) = delete;

	~VertexInputState();

	[[nodiscard]]
	DeviceSessionPtr getDeviceSession() const;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::Pipeline& getHandle();

private:
	friend class DeviceSession;

	DeviceSession* m_device_session;

	Desc m_desc;

	vk::Pipeline m_handle;
};
}

template<>
struct std::hash<cgpu::VertexInputState::Desc>
{
	std::size_t operator()(const cgpu::VertexInputState::Desc& key) const noexcept;
};
