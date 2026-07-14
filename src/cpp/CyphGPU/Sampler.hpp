#pragma once

#include <CyphGPU/fwd.hpp>
#include <CyphGPU/ShaderTypes.hpp>

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

		auto operator<=>(const Desc&) const = default;
	};

	[[nodiscard]]
	static SamplerPtr create(const DeviceSessionPtr& device_session, Desc&& desc);

	explicit Sampler(PrivateKey, DeviceSession& device_session, Desc&& desc);

	Sampler(const Sampler&) = delete;
	Sampler(Sampler&&) = delete;

	Sampler& operator=(const Sampler&) = delete;
	Sampler& operator=(Sampler&&) = delete;

	~Sampler();

	[[nodiscard]]
	DeviceSessionPtr getDeviceSession() const;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	SamplerHandle getDescriptor() const;

private:
	friend class DeviceSession;

	DeviceSession* m_device_session;

	Desc m_desc;

	uint32_t m_descriptor{};

	void createSampler();
};
}

template<>
struct std::hash<cgpu::Sampler::Desc>
{
	std::size_t operator()(const cgpu::Sampler::Desc& key) const noexcept;
};
