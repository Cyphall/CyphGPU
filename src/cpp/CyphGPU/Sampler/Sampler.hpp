#pragma once

#include <CyphGPU/fwd.hpp>

#include <vulkan/vulkan.hpp>

namespace cgpu
{
class Sampler final
{
	class PrivateKey
	{};

public:
	struct Desc
	{
		// Optional
		vk::Filter min_filter{vk::Filter::eNearest};
		vk::Filter mag_filter{vk::Filter::eNearest};
		vk::SamplerMipmapMode mipmap_mode{vk::SamplerMipmapMode::eNearest};
		vk::SamplerAddressMode wrapping_u{vk::SamplerAddressMode::eRepeat};
		vk::SamplerAddressMode wrapping_v{vk::SamplerAddressMode::eRepeat};
		vk::SamplerAddressMode wrapping_w{vk::SamplerAddressMode::eRepeat};
		std::optional<float> anisotropy{};
		std::optional<vk::CompareOp> comparison_mode{};
		float min_lod{-1000.0f};
		float max_lod{1000.0f};
		float mip_lod_bias{0.0f};
		vk::BorderColor border_color{vk::BorderColor::eFloatOpaqueBlack};
	};

	[[nodiscard]]
	static SamplerPtr create(const DeviceSessionPtr& device_session, Desc&& desc);

	explicit Sampler(PrivateKey, const DeviceSessionPtr& device_session, Desc&& desc);

	Sampler(const Sampler&) = delete;
	Sampler(Sampler&&) = delete;

	Sampler& operator=(const Sampler&) = delete;
	Sampler& operator=(Sampler&&) = delete;

	[[nodiscard]]
	const DeviceSessionPtr& getDeviceSession() const;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	uint32_t getDescriptor() const;

private:
	DeviceSessionPtr m_device_session;

	Desc m_desc;

	uint32_t m_descriptor{};

	void createSampler();
};
}
