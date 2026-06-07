#include "FragmentOutputState.hpp"

#include <CyphGPU/DeviceSession/DeviceSession.hpp>
#include <CyphGPU/Utils.hpp>

#include <vulkan/vulkan_hash.hpp>

cgpu::FragmentOutputStatePtr cgpu::FragmentOutputState::create(const DeviceSessionPtr& device_session, Desc&& desc)
{
	return {device_session->shared_from_this(), &device_session->getFragmentOutputState(std::move(desc))};
}

cgpu::FragmentOutputState::FragmentOutputState(PrivateKey, DeviceSession& device_session, Desc&& desc):
	m_device_session{&device_session},
	m_desc{std::move(desc)}
{
	vk::PipelineMultisampleStateCreateInfo multisample_state;
	multisample_state.flags;
	multisample_state.rasterizationSamples;
	multisample_state.sampleShadingEnable;
	multisample_state.minSampleShading;
	multisample_state.pSampleMask;
	multisample_state.alphaToCoverageEnable;
	multisample_state.alphaToOneEnable;

	vk::PipelineDepthStencilStateCreateInfo depth_stencil_state;
	depth_stencil_state.flags;
	depth_stencil_state.depthTestEnable;
	depth_stencil_state.depthWriteEnable;
	depth_stencil_state.depthCompareOp;
	depth_stencil_state.depthBoundsTestEnable;
	depth_stencil_state.stencilTestEnable;
	depth_stencil_state.front;
	depth_stencil_state.back;
	depth_stencil_state.minDepthBounds;
	depth_stencil_state.maxDepthBounds;

	vk::PipelineColorBlendStateCreateInfo color_blend_state;
	color_blend_state.flags;
	color_blend_state.logicOpEnable;
	color_blend_state.logicOp;
	color_blend_state.attachmentCount;
	color_blend_state.pAttachments;
	color_blend_state.blendConstants;

	vk::StructureChain<
		vk::GraphicsPipelineCreateInfo,
		vk::PipelineCreateFlags2CreateInfo,
		vk::GraphicsPipelineLibraryCreateInfoEXT,
		vk::PipelineRenderingCreateInfo>
		chain;

	auto& create_info = chain.get<vk::GraphicsPipelineCreateInfo>();
	create_info.flags = {};
	create_info.stageCount = 0;
	create_info.pStages = nullptr;
	create_info.pVertexInputState = nullptr;
	create_info.pInputAssemblyState = nullptr;
	create_info.pTessellationState = nullptr;
	create_info.pViewportState = nullptr;
	create_info.pRasterizationState = nullptr;
	create_info.pMultisampleState = &multisample_state;
	create_info.pDepthStencilState = &depth_stencil_state;
	create_info.pColorBlendState = &color_blend_state;
	create_info.pDynamicState = nullptr;
	create_info.layout = nullptr;
	create_info.renderPass = nullptr;
	create_info.subpass = 0;
	create_info.basePipelineHandle = nullptr;
	create_info.basePipelineIndex = 0;

	auto& flags_create_info = chain.get<vk::PipelineCreateFlags2CreateInfo>();
	flags_create_info.flags =
		vk::PipelineCreateFlagBits2::eDescriptorHeapEXT |
		vk::PipelineCreateFlagBits2::eRetainLinkTimeOptimizationInfoEXT |
		vk::PipelineCreateFlagBits2::eLibraryKHR;

	auto& library_create_info = chain.get<vk::GraphicsPipelineLibraryCreateInfoEXT>();
	library_create_info.flags = vk::GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface;

	auto& rendering_create_info = chain.get<vk::PipelineRenderingCreateInfo>();
	rendering_create_info.viewMask;
	rendering_create_info.colorAttachmentCount;
	rendering_create_info.pColorAttachmentFormats;
	rendering_create_info.depthAttachmentFormat;
	rendering_create_info.stencilAttachmentFormat;

	m_handle = m_device_session->getHandle().createGraphicsPipeline(nullptr, chain.get(), nullptr, m_device_session->getDispatcher()).value;
}

cgpu::FragmentOutputState::~FragmentOutputState()
{
	m_device_session->getHandle().destroy(m_handle, nullptr, m_device_session->getDispatcher());
}

cgpu::DeviceSessionPtr cgpu::FragmentOutputState::getDeviceSession() const
{
	return m_device_session->shared_from_this();
}

const cgpu::FragmentOutputState::Desc& cgpu::FragmentOutputState::getDesc() const
{
	return m_desc;
}

const vk::Pipeline& cgpu::FragmentOutputState::getHandle() const
{
	return m_handle;
}

std::size_t std::hash<cgpu::FragmentOutputState::Desc>::operator()(const cgpu::FragmentOutputState::Desc& key) const noexcept
{
	size_t seed = 0;
	cgpu::hashCombine(seed, key.topology);
	return seed;
}
