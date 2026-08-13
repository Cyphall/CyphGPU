#include "GraphicsPassContext.hpp"

#include <CyphGPU/CommandRecorder.hpp>
#include <CyphGPU/TLAS.hpp>

cgpu::SampledImageHandle cgpu::GraphicsPassContext::getSampledImageDescriptor(const ImagePtr& image, GraphicsStages stages, const Image::SampledDescriptorOverrides& overrides)
{
	registerSampledImageIndirectAccess(image, stages);
	return image->getSampledDescriptorIndirect(overrides);
}

cgpu::StorageImageHandle cgpu::GraphicsPassContext::getStorageImageDescriptor(const ImagePtr& image, GraphicsStages stages, StorageAccess access, const Image::StorageDescriptorOverrides& overrides)
{
	registerStorageImageIndirectAccess(image, stages, access);
	return image->getStorageDescriptorIndirect(overrides);
}

cgpu::UniformTexelBufferHandle cgpu::GraphicsPassContext::getUniformTexelBufferDescriptor(const BufferPtr& buffer, GraphicsStages stages, vk::Format format, const Buffer::UniformTexelDescriptorOverrides& overrides)
{
	registerSampledBufferIndirectAccess(buffer, stages);
	return buffer->getUniformTexelDescriptorIndirect(format, overrides);
}

cgpu::StorageTexelBufferHandle cgpu::GraphicsPassContext::getStorageTexelBufferDescriptor(const BufferPtr& buffer, GraphicsStages stages, StorageAccess access, vk::Format format, const Buffer::StorageTexelDescriptorOverrides& overrides)
{
	registerStorageBufferIndirectAccess(buffer, stages, access);
	return buffer->getStorageTexelDescriptorIndirect(format, overrides);
}

vk::DeviceAddress cgpu::GraphicsPassContext::getTLASDevicePtr(const TLASPtr& tlas, GraphicsStages stages)
{
	registerTLASIndirectAccess(tlas, stages);
	return tlas->getDevicePtr();
}

void cgpu::GraphicsPassContext::registerSampledImageIndirectAccess(const ImagePtr& image, GraphicsStages stages)
{
	m_rec->addCmdResource(image, toVk(stages), vk::AccessFlagBits2::eShaderSampledRead);
}

void cgpu::GraphicsPassContext::registerStorageImageIndirectAccess(const ImagePtr& image, GraphicsStages stages, StorageAccess access)
{
	m_rec->addCmdResource(image, toVk(stages), PassContext::toVk(access));
}

void cgpu::GraphicsPassContext::registerSampledBufferIndirectAccess(const BufferPtr& buffer, GraphicsStages stages)
{
	m_rec->addCmdResource(buffer, toVk(stages), vk::AccessFlagBits2::eShaderSampledRead);
}

void cgpu::GraphicsPassContext::registerStorageBufferIndirectAccess(const BufferPtr& buffer, GraphicsStages stages, StorageAccess access)
{
	m_rec->addCmdResource(buffer, toVk(stages), PassContext::toVk(access));
}

void cgpu::GraphicsPassContext::registerTLASIndirectAccess(const TLASPtr& tlas, GraphicsStages stages)
{
	m_rec->addCmdResource(tlas->getBuffer(), toVk(stages), vk::AccessFlagBits2::eAccelerationStructureReadKHR);
}

void cgpu::GraphicsPassContext::bindPipelineStates(
	const VertexInputStatePtr& vertex_input_state,
	const PreRasterizationShaderStatePtr& pre_rasterization_shader_state,
	const FragmentShaderStatePtr& fragment_shader_state,
	const FragmentOutputStatePtr& fragment_output_state
)
{
	if (vertex_input_state == m_current_vertex_input_state &&
	    pre_rasterization_shader_state == m_current_pre_rasterization_shader_state &&
	    fragment_shader_state == m_current_fragment_shader_state &&
	    fragment_output_state == m_current_fragment_output_state)
	{
		return;
	}

	m_rec->bindPipelineStates(
		vertex_input_state,
		pre_rasterization_shader_state,
		fragment_shader_state,
		fragment_output_state
	);

	m_current_vertex_input_state = vertex_input_state;
	m_current_pre_rasterization_shader_state = pre_rasterization_shader_state;
	m_current_fragment_shader_state = fragment_shader_state;
	m_current_fragment_output_state = fragment_output_state;
}

void cgpu::GraphicsPassContext::bindIndexBuffer(
	const BufferPtr& buffer,
	vk::IndexType index_type,
	std::optional<Range<vk::DeviceSize>> range
)
{
	m_rec->bindIndexBuffer(
		buffer,
		index_type,
		range ? *range : Range<vk::DeviceSize>{0, buffer->getDesc().size}
	);
}

void cgpu::GraphicsPassContext::draw(
	uint32_t vertex_count,
	uint32_t instance_count,
	uint32_t first_vertex,
	uint32_t first_instance,
	const void* data,
	size_t size,
	size_t alignment
)
{
	vk::DeviceAddress gpu_ptr = m_rec->writeParameters(
		data,
		size,
		alignment
	);

	m_rec->pushParameterPtr(gpu_ptr);

	m_rec->draw(
		vertex_count,
		instance_count,
		first_vertex,
		first_instance
	);
}

void cgpu::GraphicsPassContext::drawIndexed(
	uint32_t index_count,
	uint32_t instance_count,
	uint32_t first_index,
	int32_t vertex_offset,
	uint32_t first_instance,
	const void* data,
	size_t size,
	size_t alignment
)
{
	vk::DeviceAddress gpu_ptr = m_rec->writeParameters(
		data,
		size,
		alignment
	);

	m_rec->pushParameterPtr(gpu_ptr);

	m_rec->drawIndexed(
		index_count,
		instance_count,
		first_index,
		vertex_offset,
		first_instance
	);
}

void cgpu::GraphicsPassContext::setViewport(
	const vk::Viewport& viewport
)
{
	m_rec->setViewport(viewport);
}

void cgpu::GraphicsPassContext::setScissor(
	const vk::Rect2D& scissor
)
{
	m_rec->setScissor(scissor);
}

vk::PipelineStageFlags2 cgpu::GraphicsPassContext::toVk(GraphicsStages stages)
{
	assert(stages != GraphicsStages{});

	vk::PipelineStageFlags2 vk_stages;
	if (stages & GraphicsStage::eVertex)
	{
		vk_stages |= vk::PipelineStageFlagBits2::eVertexShader;
	}
	if (stages & GraphicsStage::eGeometry)
	{
		vk_stages |= vk::PipelineStageFlagBits2::eGeometryShader;
	}
	if (stages & GraphicsStage::eFragment)
	{
		vk_stages |= vk::PipelineStageFlagBits2::eFragmentShader;
	}
	return vk_stages;
}
