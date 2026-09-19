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
		/// Optional. Default: Nearest.
		vk::Filter min_filter{vk::Filter::eNearest};
		/// Optional. Default: Nearest.
		vk::Filter mag_filter{vk::Filter::eNearest};
		/// Optional. Default: Nearest.
		vk::SamplerMipmapMode mipmap_mode{vk::SamplerMipmapMode::eNearest};
		/// Optional. Default: Repeat.
		vk::SamplerAddressMode wrapping_u{vk::SamplerAddressMode::eRepeat};
		/// Optional. Default: Repeat.
		vk::SamplerAddressMode wrapping_v{vk::SamplerAddressMode::eRepeat};
		/// Optional. Default: Repeat.
		vk::SamplerAddressMode wrapping_w{vk::SamplerAddressMode::eRepeat};
		/// Optional. Default: No anisotropy.
		std::optional<float> anisotropy{};
		/// Optional. Default: No comparison.
		std::optional<vk::CompareOp> comparison_mode{};
		/// Optional. Default: -1000.
		float min_lod{-1000.0f};
		/// Optional. Default: 1000.
		float max_lod{1000.0f};
		/// Optional. Default: 0.
		float mip_lod_bias{0.0f};
		/// Optional. Default: Float opaque black.
		///
		/// Must be set when sampling non-float images.
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
