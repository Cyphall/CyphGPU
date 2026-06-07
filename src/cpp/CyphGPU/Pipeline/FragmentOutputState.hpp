#pragma once

#include <CyphGPU/fwd.hpp>

#include <vulkan/vulkan.hpp>

namespace cgpu
{
class FragmentOutputState final
{
	class PrivateKey
	{};

public:
	struct Desc
	{
		struct ColorAttachment
		{
		};

		// Optional
		vk::PrimitiveTopology topology{vk::PrimitiveTopology::eTriangleList};

		bool operator==(const Desc& other) const = default;
	};

	[[nodiscard]]
	static FragmentOutputStatePtr create(const DeviceSessionPtr& device_session, Desc&& desc);

	explicit FragmentOutputState(PrivateKey, DeviceSession& device_session, Desc&& desc);

	FragmentOutputState(const FragmentOutputState&) = delete;
	FragmentOutputState(FragmentOutputState&&) = delete;

	FragmentOutputState& operator=(const FragmentOutputState&) = delete;
	FragmentOutputState& operator=(FragmentOutputState&&) = delete;

	~FragmentOutputState();

	[[nodiscard]]
	DeviceSessionPtr getDeviceSession() const;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::Pipeline& getHandle() const;

private:
	friend class DeviceSession;

	DeviceSession* m_device_session;

	Desc m_desc;

	vk::Pipeline m_handle;
};
}

template<>
struct std::hash<cgpu::FragmentOutputState::Desc>
{
	std::size_t operator()(const cgpu::FragmentOutputState::Desc& key) const noexcept;
};
