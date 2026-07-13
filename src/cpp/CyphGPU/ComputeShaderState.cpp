#include "ComputeShaderState.hpp"

#include <CyphGPU/detail/ShaderChainBuilder.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/HashExt.hpp>

#include <vulkan/vulkan_hash.hpp>

cgpu::ComputeShaderStatePtr cgpu::ComputeShaderState::create(const DeviceSessionPtr& device_session, Desc&& desc)
{
	return {device_session->shared_from_this(), &device_session->getComputeShaderState(std::move(desc))};
}

cgpu::ComputeShaderState::ComputeShaderState(PrivateKey, DeviceSession& device_session, Desc&& desc):
	m_device_session{&device_session},
	m_desc{std::move(desc)}
{
	createPipelineState();
}

cgpu::ComputeShaderState::~ComputeShaderState()
{
	m_device_session->getHandle().destroy(m_handle, nullptr, m_device_session->getDispatcher());
}

cgpu::DeviceSessionPtr cgpu::ComputeShaderState::getDeviceSession() const
{
	return m_device_session->shared_from_this();
}

const cgpu::ComputeShaderState::Desc& cgpu::ComputeShaderState::getDesc() const
{
	return m_desc;
}

const vk::Pipeline& cgpu::ComputeShaderState::getHandle()
{
	return m_handle;
}

void cgpu::ComputeShaderState::createPipelineState()
{
	detail::ShaderChainBuilder shader_chain_builder{*m_device_session};
	shader_chain_builder.addShader(m_desc.compute_shader.source, m_desc.compute_shader.entry_point.c_str(), vk::ShaderStageFlagBits::eCompute);

	std::vector<vk::PipelineShaderStageCreateInfo> shader_stages = shader_chain_builder.finalize();

	vk::StructureChain<
		vk::ComputePipelineCreateInfo,
		vk::PipelineCreateFlags2CreateInfo>
		chain;

	auto& create_info = chain.get<vk::ComputePipelineCreateInfo>();
	// create_info.flags;
	create_info.stage = shader_stages[0];
	create_info.layout = nullptr;
	// create_info.basePipelineHandle;
	// create_info.basePipelineIndex;

	auto& flags_create_info = chain.get<vk::PipelineCreateFlags2CreateInfo>();
	flags_create_info.flags =
		vk::PipelineCreateFlagBits2::eDescriptorHeapEXT;

	m_handle = m_device_session->getHandle().createComputePipeline(nullptr, chain.get(), nullptr, m_device_session->getDispatcher()).value;
}

std::size_t std::hash<cgpu::ComputeShaderState::Desc>::operator()(const cgpu::ComputeShaderState::Desc& key) const noexcept
{
	size_t seed = 0;
	cgpu::hashCombine(seed, key.compute_shader);
	return seed;
}

std::size_t std::hash<cgpu::ComputeShaderState::Desc::ComputeShader>::operator()(const cgpu::ComputeShaderState::Desc::ComputeShader& key) const noexcept
{
	size_t seed = 0;
	cgpu::hashCombine(seed, key.source);
	cgpu::hashCombine(seed, key.entry_point);
	return seed;
}
