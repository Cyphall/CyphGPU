#include "FragmentOutputState.hpp"

#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/HashExt.hpp>

#include <glm/gtx/hash.hpp>
#include <vulkan/vulkan_hash.hpp>

cgpu::FragmentOutputStatePtr cgpu::FragmentOutputState::create(const DeviceSessionPtr& device_session, Desc&& desc)
{
	return {device_session->shared_from_this(), &device_session->getFragmentOutputState(std::move(desc))};
}

cgpu::FragmentOutputState::FragmentOutputState(PrivateKey, DeviceSession& device_session, Desc&& desc):
	m_device_session{&device_session},
	m_desc{std::move(desc)}
{
	createPipelineState();
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

void cgpu::FragmentOutputState::createPipelineState()
{
	vk::PipelineMultisampleStateCreateInfo multisample_state;
	multisample_state.flags = {};
	multisample_state.rasterizationSamples = m_desc.samples;
	multisample_state.sampleShadingEnable = vk::False;
	// multisample_state.minSampleShading;
	// multisample_state.pSampleMask;
	multisample_state.alphaToCoverageEnable = vk::False;
	multisample_state.alphaToOneEnable = vk::False;

	std::vector<vk::PipelineColorBlendAttachmentState> blend_states;
	blend_states.reserve(m_desc.color_attachments.size());
	for (const auto& attachment : m_desc.color_attachments)
	{
		auto& blend_state = blend_states.emplace_back();
		if (attachment.blend)
		{
			blend_state.blendEnable = vk::True;
			blend_state.srcColorBlendFactor = attachment.blend->color.src_factor;
			blend_state.dstColorBlendFactor = attachment.blend->color.dst_factor;
			blend_state.colorBlendOp = attachment.blend->color.op;
			blend_state.srcAlphaBlendFactor = attachment.blend->alpha.src_factor;
			blend_state.dstAlphaBlendFactor = attachment.blend->alpha.dst_factor;
			blend_state.alphaBlendOp = attachment.blend->alpha.op;
		}
		else
		{
			blend_state.blendEnable = vk::False;
			// blend_state.srcColorBlendFactor;
			// blend_state.dstColorBlendFactor;
			// blend_state.colorBlendOp;
			// blend_state.srcAlphaBlendFactor;
			// blend_state.dstAlphaBlendFactor;
			// blend_state.alphaBlendOp;
		}

		blend_state.colorWriteMask = attachment.write_mask;
	}

	vk::PipelineColorBlendStateCreateInfo color_blend_state;
	color_blend_state.flags = {};
	color_blend_state.logicOpEnable = vk::False;
	// color_blend_state.logicOp;
	color_blend_state.attachmentCount = static_cast<uint32_t>(blend_states.size());
	color_blend_state.pAttachments = blend_states.data();
	color_blend_state.blendConstants[0] = m_desc.blend_constants[0];
	color_blend_state.blendConstants[1] = m_desc.blend_constants[1];
	color_blend_state.blendConstants[2] = m_desc.blend_constants[2];
	color_blend_state.blendConstants[3] = m_desc.blend_constants[3];

	std::vector<vk::Format> color_attachment_formats;
	color_attachment_formats.reserve(m_desc.color_attachments.size());
	for (const auto& attachment : m_desc.color_attachments)
	{
		color_attachment_formats.emplace_back(attachment.format);
	}

	vk::StructureChain<
		vk::GraphicsPipelineCreateInfo,
		vk::PipelineCreateFlags2CreateInfo,
		vk::GraphicsPipelineLibraryCreateInfoEXT,
		vk::PipelineRenderingCreateInfo>
		chain;

	auto& create_info = chain.get<vk::GraphicsPipelineCreateInfo>();
	// create_info.flags;
	// create_info.stageCount;
	// create_info.pStages;
	// create_info.pVertexInputState;
	// create_info.pInputAssemblyState;
	// create_info.pTessellationState;
	// create_info.pViewportState;
	// create_info.pRasterizationState;
	create_info.pMultisampleState = &multisample_state;
	// create_info.pDepthStencilState;
	create_info.pColorBlendState = &color_blend_state;
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
	library_create_info.flags = vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface;

	auto& rendering_create_info = chain.get<vk::PipelineRenderingCreateInfo>();
	// rendering_create_info.viewMask;
	rendering_create_info.colorAttachmentCount = static_cast<uint32_t>(color_attachment_formats.size());
	rendering_create_info.pColorAttachmentFormats = color_attachment_formats.data();
	rendering_create_info.depthAttachmentFormat = m_desc.depth_attachment.value_or(vk::Format::eUndefined);
	rendering_create_info.stencilAttachmentFormat = m_desc.stencil_attachment.value_or(vk::Format::eUndefined);

	m_handle = m_device_session->getHandle().createGraphicsPipeline(nullptr, chain.get(), nullptr, m_device_session->getDispatcher()).value;
}

std::size_t std::hash<cgpu::FragmentOutputState::Desc>::operator()(const cgpu::FragmentOutputState::Desc& key) const noexcept
{
	size_t seed = 0;
	cgpu::hashCombine(seed, key.color_attachments);
	cgpu::hashCombine(seed, key.depth_attachment);
	cgpu::hashCombine(seed, key.stencil_attachment);
	cgpu::hashCombine(seed, key.samples);
	cgpu::hashCombine(seed, key.blend_constants);
	return seed;
}

std::size_t std::hash<cgpu::FragmentOutputState::Desc::BlendComponentState>::operator()(const cgpu::FragmentOutputState::Desc::BlendComponentState& key) const noexcept
{
	size_t seed = 0;
	cgpu::hashCombine(seed, key.src_factor);
	cgpu::hashCombine(seed, key.dst_factor);
	cgpu::hashCombine(seed, key.op);
	return seed;
}

std::size_t std::hash<cgpu::FragmentOutputState::Desc::BlendState>::operator()(const cgpu::FragmentOutputState::Desc::BlendState& key) const noexcept
{
	size_t seed = 0;
	cgpu::hashCombine(seed, key.color);
	cgpu::hashCombine(seed, key.alpha);
	return seed;
}

std::size_t std::hash<cgpu::FragmentOutputState::Desc::ColorAttachment>::operator()(const cgpu::FragmentOutputState::Desc::ColorAttachment& key) const noexcept
{
	size_t seed = 0;
	cgpu::hashCombine(seed, key.format);
	cgpu::hashCombine(seed, key.blend);
	cgpu::hashCombine(seed, key.write_mask);
	return seed;
}
