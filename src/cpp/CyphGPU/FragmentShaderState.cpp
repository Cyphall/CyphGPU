#include "FragmentShaderState.hpp"

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

cgpu::FragmentShaderStatePtr cgpu::FragmentShaderState::create(const DeviceSessionPtr& device_session, Desc&& desc)
{
	return {device_session->shared_from_this(), &device_session->getFragmentShaderState(std::move(desc))};
}

cgpu::FragmentShaderState::FragmentShaderState(PrivateKey, DeviceSession& device_session, Desc&& desc):
	m_device_session{&device_session},
	m_desc{std::move(desc)}
{
	createPipelineState();
}

cgpu::FragmentShaderState::~FragmentShaderState()
{
	m_device_session->getHandle().destroy(m_handle, nullptr, m_device_session->getDispatcher());
}

cgpu::DeviceSessionPtr cgpu::FragmentShaderState::getDeviceSession() const
{
	return m_device_session->shared_from_this();
}

const cgpu::FragmentShaderState::Desc& cgpu::FragmentShaderState::getDesc() const
{
	return m_desc;
}

const vk::Pipeline& cgpu::FragmentShaderState::getHandle()
{
	return m_handle;
}

void cgpu::FragmentShaderState::createPipelineState()
{
	std::vector<ShaderChain> shader_chains;
	if (m_desc.fragment_shader)
	{
		shader_chains.emplace_back(makeShaderChain(
			m_desc.fragment_shader->blob,
			m_desc.fragment_shader->entry_point.c_str(),
			vk::ShaderStageFlagBits::eFragment,
			m_device_session->getMappings()
		));
	}

	std::vector<vk::PipelineShaderStageCreateInfo> packed_shader_infos;
	for (const auto& chain : shader_chains)
	{
		packed_shader_infos.emplace_back(chain.get());
	}

	vk::PipelineDepthStencilStateCreateInfo depth_stencil_state;
	depth_stencil_state.flags = {};
	if (m_desc.depth_state)
	{
		depth_stencil_state.depthTestEnable = vk::True;
		depth_stencil_state.depthWriteEnable = m_desc.depth_state->write_enabled ? vk::True : vk::False;
		depth_stencil_state.depthCompareOp = m_desc.depth_state->test_pass_condition;
	}
	else
	{
		depth_stencil_state.depthTestEnable = vk::False;
		// depth_stencil_state.depthWriteEnable;
		// depth_stencil_state.depthCompareOp;
	}
	depth_stencil_state.depthBoundsTestEnable = vk::False;
	if (m_desc.stencil_state)
	{
		depth_stencil_state.stencilTestEnable = vk::True;
		depth_stencil_state.front = m_desc.stencil_state->front;
		depth_stencil_state.back = m_desc.stencil_state->front;
	}
	else
	{
		depth_stencil_state.stencilTestEnable = vk::False;
		// depth_stencil_state.front;
		// depth_stencil_state.back;
	}
	// depth_stencil_state.minDepthBounds;
	// depth_stencil_state.maxDepthBounds;

	vk::StructureChain<
		vk::GraphicsPipelineCreateInfo,
		vk::PipelineCreateFlags2CreateInfo,
		vk::GraphicsPipelineLibraryCreateInfoEXT,
		vk::PipelineRenderingCreateInfo>
		chain;

	auto& create_info = chain.get<vk::GraphicsPipelineCreateInfo>();
	create_info.flags = {};
	create_info.stageCount = static_cast<uint32_t>(packed_shader_infos.size());
	create_info.pStages = packed_shader_infos.data();
	// create_info.pVertexInputState;
	// create_info.pInputAssemblyState;
	// create_info.pTessellationState;
	// create_info.pViewportState;
	// create_info.pRasterizationState;
	// create_info.pMultisampleState;
	create_info.pDepthStencilState = &depth_stencil_state;
	// create_info.pColorBlendState;
	// create_info.pDynamicState;
	// create_info.layout;
	create_info.renderPass = nullptr;
	// create_info.subpass;
	// create_info.basePipelineHandle;
	// create_info.basePipelineIndex;

	auto& flags_create_info = chain.get<vk::PipelineCreateFlags2CreateInfo>();
	flags_create_info.flags =
		vk::PipelineCreateFlagBits2::eDescriptorHeapEXT |
		vk::PipelineCreateFlagBits2::eRetainLinkTimeOptimizationInfoEXT |
		vk::PipelineCreateFlagBits2::eLibraryKHR;

	auto& library_create_info = chain.get<vk::GraphicsPipelineLibraryCreateInfoEXT>();
	library_create_info.flags = vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader;

	auto& rendering_create_info = chain.get<vk::PipelineRenderingCreateInfo>();
	rendering_create_info.viewMask = m_desc.view_mask;
	// rendering_create_info.colorAttachmentCount;
	// rendering_create_info.pColorAttachmentFormats;
	// rendering_create_info.depthAttachmentFormat;
	// rendering_create_info.stencilAttachmentFormat;

	m_handle = m_device_session->getHandle().createGraphicsPipeline(nullptr, chain.get(), nullptr, m_device_session->getDispatcher()).value;
}

std::size_t std::hash<cgpu::FragmentShaderState::Desc>::operator()(const cgpu::FragmentShaderState::Desc& key) const noexcept
{
	size_t seed = 0;
	cgpu::hashCombine(seed, key.fragment_shader);
	cgpu::hashCombine(seed, key.depth_state);
	cgpu::hashCombine(seed, key.stencil_state);
	cgpu::hashCombine(seed, key.view_mask);
	return seed;
}

std::size_t std::hash<cgpu::FragmentShaderState::Desc::FragmentShader>::operator()(const cgpu::FragmentShaderState::Desc::FragmentShader& key) const noexcept
{
	size_t seed = 0;
	cgpu::hashCombine(seed, key.blob);
	cgpu::hashCombine(seed, key.entry_point);
	return seed;
}

std::size_t std::hash<cgpu::FragmentShaderState::Desc::DepthState>::operator()(const cgpu::FragmentShaderState::Desc::DepthState& key) const noexcept
{
	size_t seed = 0;
	cgpu::hashCombine(seed, key.test_pass_condition);
	cgpu::hashCombine(seed, key.write_enabled);
	return seed;
}

std::size_t std::hash<cgpu::FragmentShaderState::Desc::StencilState>::operator()(const cgpu::FragmentShaderState::Desc::StencilState& key) const noexcept
{
	size_t seed = 0;
	cgpu::hashCombine(seed, key.front);
	cgpu::hashCombine(seed, key.back);
	return seed;
}
