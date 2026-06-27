#include "PreRasterizationShaderState.hpp"

#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/HashExt.hpp>

#include <vulkan/vulkan_hash.hpp>

namespace
{
using ShaderChain = vk::StructureChain<
	vk::PipelineShaderStageCreateInfo,
	vk::ShaderModuleCreateInfo,
	vk::ShaderDescriptorSetAndBindingMappingInfoEXT>;

[[nodiscard]]
ShaderChain makeShaderChain(std::span<const uint32_t> blob, const char* entry_point, vk::ShaderStageFlagBits stage, std::span<const vk::DescriptorSetAndBindingMappingEXT> mappings)
{
	ShaderChain chain;

	auto& stage_info = chain.get<vk::PipelineShaderStageCreateInfo>();
	stage_info.flags = {};
	stage_info.stage = stage;
	stage_info.module = nullptr;
	stage_info.pName = entry_point;
	stage_info.pSpecializationInfo = nullptr;

	auto& module_info = chain.get<vk::ShaderModuleCreateInfo>();
	module_info.flags = {};
	module_info.codeSize = static_cast<uint32_t>(blob.size() * sizeof(uint32_t));
	module_info.pCode = blob.data();

	auto& mapping_info = chain.get<vk::ShaderDescriptorSetAndBindingMappingInfoEXT>();
	mapping_info.mappingCount = static_cast<uint32_t>(mappings.size());
	mapping_info.pMappings = mappings.data();

	return chain;
}
}

cgpu::PreRasterizationShaderStatePtr cgpu::PreRasterizationShaderState::create(const DeviceSessionPtr& device_session, Desc&& desc)
{
	return {device_session->shared_from_this(), &device_session->getPreRasterizationShaderState(std::move(desc))};
}

cgpu::PreRasterizationShaderState::PreRasterizationShaderState(PrivateKey, DeviceSession& device_session, Desc&& desc):
	m_device_session{&device_session},
	m_desc{std::move(desc)}
{
	createPipelineState();
}

cgpu::PreRasterizationShaderState::~PreRasterizationShaderState()
{
	m_device_session->getHandle().destroy(m_handle, nullptr, m_device_session->getDispatcher());
}

cgpu::DeviceSessionPtr cgpu::PreRasterizationShaderState::getDeviceSession() const
{
	return m_device_session->shared_from_this();
}

const cgpu::PreRasterizationShaderState::Desc& cgpu::PreRasterizationShaderState::getDesc() const
{
	return m_desc;
}

const vk::Pipeline& cgpu::PreRasterizationShaderState::getHandle()
{
	return m_handle;
}

void cgpu::PreRasterizationShaderState::createPipelineState()
{
	std::vector<ShaderChain> shader_chains;
	shader_chains.emplace_back(makeShaderChain(
		m_desc.vertex_shader.blob,
		m_desc.vertex_shader.entry_point.c_str(),
		vk::ShaderStageFlagBits::eVertex,
		m_device_session->getMappings()
	));
	if (m_desc.geometry_shader)
	{
		shader_chains.emplace_back(makeShaderChain(
			m_desc.geometry_shader->blob,
			m_desc.geometry_shader->entry_point.c_str(),
			vk::ShaderStageFlagBits::eGeometry,
			m_device_session->getMappings()
		));
	}

	std::vector<vk::PipelineShaderStageCreateInfo> packed_shader_infos;
	for (const auto& chain : shader_chains)
	{
		packed_shader_infos.emplace_back(chain.get());
	}

	vk::PipelineViewportStateCreateInfo viewport_state;
	viewport_state.flags = {};
	viewport_state.viewportCount = 1;
	// viewport_state.pViewports;
	viewport_state.scissorCount = 1;
	// viewport_state.pScissors;

	vk::PipelineRasterizationStateCreateInfo rasterization_state;
	rasterization_state.flags = {};
	rasterization_state.depthClampEnable = m_desc.depth_clamp ? vk::True : vk::False;
	rasterization_state.rasterizerDiscardEnable = vk::False;
	rasterization_state.polygonMode = m_desc.polygon_mode;
	rasterization_state.cullMode = m_desc.cull_mode;
	rasterization_state.frontFace = m_desc.front_face;
	rasterization_state.depthBiasEnable = vk::False;
	// rasterization_state.depthBiasConstantFactor;
	// rasterization_state.depthBiasClamp;
	// rasterization_state.depthBiasSlopeFactor;
	rasterization_state.lineWidth = m_desc.line_width;

	std::array dynamic_states = {
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor,
	};

	vk::PipelineDynamicStateCreateInfo dynamic_state;
	dynamic_state.flags = {};
	dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
	dynamic_state.pDynamicStates = dynamic_states.data();

	vk::StructureChain<
		vk::GraphicsPipelineCreateInfo,
		vk::PipelineCreateFlags2CreateInfo,
		vk::GraphicsPipelineLibraryCreateInfoEXT>
		chain;

	auto& create_info = chain.get<vk::GraphicsPipelineCreateInfo>();
	create_info.flags = {};
	create_info.stageCount = static_cast<uint32_t>(packed_shader_infos.size());
	create_info.pStages = packed_shader_infos.data();
	// create_info.pVertexInputState;
	// create_info.pInputAssemblyState;
	create_info.pTessellationState = nullptr;
	create_info.pViewportState = &viewport_state;
	create_info.pRasterizationState = &rasterization_state;
	// create_info.pMultisampleState;
	// create_info.pDepthStencilState;
	// create_info.pColorBlendState;
	create_info.pDynamicState = &dynamic_state;
	// create_info.layout;
	// create_info.renderPass;
	// create_info.subpass;
	// create_info.basePipelineHandle;
	// create_info.basePipelineIndex;

	auto& flags_create_info = chain.get<vk::PipelineCreateFlags2CreateInfo>();
	flags_create_info.flags =
		vk::PipelineCreateFlagBits2::eDescriptorHeapEXT |
		vk::PipelineCreateFlagBits2::eRetainLinkTimeOptimizationInfoEXT |
		vk::PipelineCreateFlagBits2::eLibraryKHR;

	auto& library_create_info = chain.get<vk::GraphicsPipelineLibraryCreateInfoEXT>();
	library_create_info.flags = vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders;

	m_handle = m_device_session->getHandle().createGraphicsPipeline(nullptr, chain.get(), nullptr, m_device_session->getDispatcher()).value;
}

std::size_t std::hash<cgpu::PreRasterizationShaderState::Desc>::operator()(const cgpu::PreRasterizationShaderState::Desc& key) const noexcept
{
	size_t seed = 0;
	cgpu::hashCombine(seed, key.vertex_shader);
	cgpu::hashCombine(seed, key.geometry_shader);
	cgpu::hashCombine(seed, key.depth_clamp);
	cgpu::hashCombine(seed, key.polygon_mode);
	cgpu::hashCombine(seed, key.cull_mode);
	cgpu::hashCombine(seed, key.front_face);
	cgpu::hashCombine(seed, key.line_width);
	cgpu::hashCombine(seed, key.view_mask);
	return seed;
}

std::size_t std::hash<cgpu::PreRasterizationShaderState::Desc::VertexShader>::operator()(const cgpu::PreRasterizationShaderState::Desc::VertexShader& key) const noexcept
{
	size_t seed = 0;
	cgpu::hashCombine(seed, key.blob);
	cgpu::hashCombine(seed, key.entry_point);
	return seed;
}

std::size_t std::hash<cgpu::PreRasterizationShaderState::Desc::GeometryShader>::operator()(const cgpu::PreRasterizationShaderState::Desc::GeometryShader& key) const noexcept
{
	size_t seed = 0;
	cgpu::hashCombine(seed, key.blob);
	cgpu::hashCombine(seed, key.entry_point);
	return seed;
}
