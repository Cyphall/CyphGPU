#include "VertexInputState.hpp"

#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/Utils.hpp>

#include <vulkan/vulkan_hash.hpp>

cgpu::VertexInputStatePtr cgpu::VertexInputState::create(const DeviceSessionPtr& device_session, Desc&& desc)
{
	return {device_session->shared_from_this(), &device_session->getVertexInputState(std::move(desc))};
}

cgpu::VertexInputState::VertexInputState(PrivateKey, DeviceSession& device_session, Desc&& desc):
	m_device_session{&device_session},
	m_desc{std::move(desc)}
{
	vk::PipelineVertexInputStateCreateInfo vertex_input_state;
	vertex_input_state.flags = {};
	vertex_input_state.vertexBindingDescriptionCount = 0;
	vertex_input_state.pVertexBindingDescriptions = nullptr;
	vertex_input_state.vertexAttributeDescriptionCount = 0;
	vertex_input_state.pVertexAttributeDescriptions = nullptr;

	vk::PipelineInputAssemblyStateCreateInfo input_assembly_state;
	input_assembly_state.flags = {};
	input_assembly_state.topology = m_desc.topology;
	input_assembly_state.primitiveRestartEnable = vk::False;

	vk::StructureChain<
		vk::GraphicsPipelineCreateInfo,
		vk::PipelineCreateFlags2CreateInfo,
		vk::GraphicsPipelineLibraryCreateInfoEXT>
		chain;

	auto& create_info = chain.get<vk::GraphicsPipelineCreateInfo>();
	create_info.flags = {};
	create_info.stageCount = 0;
	create_info.pStages = nullptr;
	create_info.pVertexInputState = &vertex_input_state;
	create_info.pInputAssemblyState = &input_assembly_state;
	create_info.pTessellationState = nullptr;
	create_info.pViewportState = nullptr;
	create_info.pRasterizationState = nullptr;
	create_info.pMultisampleState = nullptr;
	create_info.pDepthStencilState = nullptr;
	create_info.pColorBlendState = nullptr;
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

	m_handle = m_device_session->getHandle().createGraphicsPipeline(nullptr, chain.get(), nullptr, m_device_session->getDispatcher()).value;
}

cgpu::VertexInputState::~VertexInputState()
{
	m_device_session->getHandle().destroy(m_handle, nullptr, m_device_session->getDispatcher());
}

cgpu::DeviceSessionPtr cgpu::VertexInputState::getDeviceSession() const
{
	return m_device_session->shared_from_this();
}

const cgpu::VertexInputState::Desc& cgpu::VertexInputState::getDesc() const
{
	return m_desc;
}

const vk::Pipeline& cgpu::VertexInputState::getHandle()
{
	return m_handle;
}

std::size_t std::hash<cgpu::VertexInputState::Desc>::operator()(const cgpu::VertexInputState::Desc& key) const noexcept
{
	size_t seed = 0;
	cgpu::hashCombine(seed, key.topology);
	return seed;
}
