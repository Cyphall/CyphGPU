#include "ComputeShaderState.hpp"

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
	vk::StructureChain<
		vk::PipelineShaderStageCreateInfo,
		vk::ShaderModuleCreateInfo,
		vk::ShaderDescriptorSetAndBindingMappingInfoEXT>
		shader_chain;

	auto& stage_info = shader_chain.get<vk::PipelineShaderStageCreateInfo>();
	stage_info.flags = {};
	stage_info.stage = vk::ShaderStageFlagBits::eCompute;
	stage_info.module = nullptr;
	stage_info.pName = m_desc.compute_shader.entry_point.c_str();
	stage_info.pSpecializationInfo = nullptr;

	auto& module_info = shader_chain.get<vk::ShaderModuleCreateInfo>();
	module_info.flags = {};
	module_info.codeSize = static_cast<uint32_t>(m_desc.compute_shader.blob.size() * sizeof(uint32_t));
	module_info.pCode = m_desc.compute_shader.blob.data();

	auto& mapping_info = shader_chain.get<vk::ShaderDescriptorSetAndBindingMappingInfoEXT>();
	mapping_info.mappingCount = static_cast<uint32_t>(m_device_session->getMappings().size());
	mapping_info.pMappings = m_device_session->getMappings().data();

	vk::StructureChain<
		vk::ComputePipelineCreateInfo,
		vk::PipelineCreateFlags2CreateInfo>
		chain;

	auto& create_info = chain.get<vk::ComputePipelineCreateInfo>();
	// create_info.flags;
	create_info.stage = shader_chain.get();
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
	cgpu::hashCombine(seed, key.blob);
	cgpu::hashCombine(seed, key.entry_point);
	return seed;
}
