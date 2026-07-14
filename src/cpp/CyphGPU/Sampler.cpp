#include "Sampler.hpp"

#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/HashExt.hpp>

#include <vulkan/vulkan_hash.hpp>

cgpu::SamplerPtr cgpu::Sampler::create(const DeviceSessionPtr& device_session, Desc&& desc)
{
	return {device_session->shared_from_this(), &device_session->getSampler(std::move(desc))};
}

cgpu::Sampler::Sampler(PrivateKey, DeviceSession& device_session, Desc&& desc):
	m_device_session{&device_session},
	m_desc{std::move(desc)}
{
	createSampler();
}

cgpu::Sampler::~Sampler()
{
	m_device_session->deleteSamplerDescriptor(m_descriptor);
}

cgpu::DeviceSessionPtr cgpu::Sampler::getDeviceSession() const
{
	return m_device_session->shared_from_this();
}

const cgpu::Sampler::Desc& cgpu::Sampler::getDesc() const
{
	return m_desc;
}

cgpu::SamplerHandle cgpu::Sampler::getDescriptor() const
{
	return m_descriptor;
}

void cgpu::Sampler::createSampler()
{
	vk::SamplerCreateInfo sampler_info;
	sampler_info.flags = {};
	sampler_info.magFilter = m_desc.mag_filter;
	sampler_info.minFilter = m_desc.min_filter;
	sampler_info.mipmapMode = m_desc.mipmap_mode;
	sampler_info.addressModeU = m_desc.wrapping_u;
	sampler_info.addressModeV = m_desc.wrapping_v;
	sampler_info.addressModeW = m_desc.wrapping_w;
	sampler_info.mipLodBias = m_desc.mip_lod_bias;
	if (m_desc.anisotropy)
	{
		sampler_info.anisotropyEnable = vk::True;
		sampler_info.maxAnisotropy = *m_desc.anisotropy;
	}
	else
	{
		sampler_info.anisotropyEnable = vk::False;
		// sampler_info.maxAnisotropy;
	}
	if (m_desc.comparison_mode)
	{
		sampler_info.compareEnable = vk::True;
		sampler_info.compareOp = *m_desc.comparison_mode;
	}
	else
	{
		sampler_info.compareEnable = vk::False;
		// sampler_info.compareOp;
	}
	sampler_info.minLod = m_desc.min_lod;
	sampler_info.maxLod = m_desc.max_lod;
	sampler_info.borderColor = m_desc.border_color;
	sampler_info.unnormalizedCoordinates = vk::False;

	m_descriptor = m_device_session->createSamplerDescriptor(sampler_info);
}

std::size_t std::hash<cgpu::Sampler::Desc>::operator()(const cgpu::Sampler::Desc& key) const noexcept
{
	size_t seed = 0;
	cgpu::hashCombine(seed, key.min_filter);
	cgpu::hashCombine(seed, key.mag_filter);
	cgpu::hashCombine(seed, key.mipmap_mode);
	cgpu::hashCombine(seed, key.wrapping_u);
	cgpu::hashCombine(seed, key.wrapping_v);
	cgpu::hashCombine(seed, key.wrapping_w);
	cgpu::hashCombine(seed, key.anisotropy);
	cgpu::hashCombine(seed, key.comparison_mode);
	cgpu::hashCombine(seed, key.min_lod);
	cgpu::hashCombine(seed, key.max_lod);
	cgpu::hashCombine(seed, key.mip_lod_bias);
	cgpu::hashCombine(seed, key.border_color);
	return seed;
}
