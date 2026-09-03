#include "GraphicsPassContext.hpp"

#include <CyphGPU/CommandRecorder.hpp>
#include <CyphGPU/DeviceSession.hpp>
#include <CyphGPU/TLAS.hpp>

#include <tracy/Tracy.hpp>

#define PROFILE_COMMANDS

#if defined(PROFILE_COMMANDS)
#	define COMMAND ZoneScoped;
#	define VULKAN_CALL(name) ZoneScopedN(#name);
#else
#	define COMMAND
#	define VULKAN_CALL(name)
#endif

cgpu::SampledImageHandle cgpu::GraphicsPassContext::getSampledImageDescriptor(
	const ImagePtr& image,
	GraphicsStages stages,
	const SampledImageDescriptorOverrides& overrides
)
{
	registerSampledImageIndirectAccess(image, stages);
	return image->getSampledDescriptorIndirect(overrides);
}

cgpu::StorageImageHandle cgpu::GraphicsPassContext::getStorageImageDescriptor(
	const ImagePtr& image,
	GraphicsStages stages,
	StorageAccess access,
	const StorageImageDescriptorOverrides& overrides
)
{
	registerStorageImageIndirectAccess(image, stages, access);
	return image->getStorageDescriptorIndirect(overrides);
}

cgpu::UniformTexelBufferHandle cgpu::GraphicsPassContext::getUniformTexelBufferDescriptor(
	const BufferPtr& buffer,
	GraphicsStages stages,
	vk::Format format,
	const UniformTexelBufferDescriptorOverrides& overrides
)
{
	registerSampledBufferIndirectAccess(buffer, stages);
	return buffer->getUniformTexelDescriptorIndirect(format, overrides);
}

cgpu::StorageTexelBufferHandle cgpu::GraphicsPassContext::getStorageTexelBufferDescriptor(
	const BufferPtr& buffer,
	GraphicsStages stages,
	StorageAccess access,
	vk::Format format,
	const StorageTexelBufferDescriptorOverrides& overrides
)
{
	registerStorageBufferIndirectAccess(buffer, stages, access);
	return buffer->getStorageTexelDescriptorIndirect(format, overrides);
}

vk::DeviceAddress cgpu::GraphicsPassContext::getTLASDevicePtr(
	const TLASPtr& tlas,
	GraphicsStages stages
)
{
	registerTLASIndirectAccess(tlas, stages);
	return tlas->getDevicePtr();
}

void cgpu::GraphicsPassContext::registerSampledImageIndirectAccess(const ImagePtr& image, GraphicsStages stages)
{
	m_rec->addCmdResource(
		image,
		{
			toVk(stages),
			vk::AccessFlagBits2::eShaderSampledRead,
		}
	);
}

void cgpu::GraphicsPassContext::registerStorageImageIndirectAccess(const ImagePtr& image, GraphicsStages stages, StorageAccess access)
{
	m_rec->addCmdResource(
		image,
		{
			toVk(stages),
			PassContext::toVk(access),
		}
	);
}

void cgpu::GraphicsPassContext::registerSampledBufferIndirectAccess(const BufferPtr& buffer, GraphicsStages stages)
{
	m_rec->addCmdResource(
		buffer,
		{
			toVk(stages),
			vk::AccessFlagBits2::eShaderSampledRead,
		}
	);
}

void cgpu::GraphicsPassContext::registerStorageBufferIndirectAccess(const BufferPtr& buffer, GraphicsStages stages, StorageAccess access)
{
	m_rec->addCmdResource(
		buffer,
		{
			toVk(stages),
			PassContext::toVk(access),
		}
	);
}

void cgpu::GraphicsPassContext::registerTLASIndirectAccess(const TLASPtr& tlas, GraphicsStages stages)
{
	m_rec->addCmdResource(
		tlas->getBuffer(),
		{
			toVk(stages),
			vk::AccessFlagBits2::eAccelerationStructureReadKHR,
		}
	);
	m_rec->addReferencedObject(tlas);
}

void cgpu::GraphicsPassContext::bindPipelineStates(
	const VertexInputStatePtr& vertex_input_state,
	const PreRasterizationShaderStatePtr& pre_rasterization_shader_state,
	const FragmentShaderStatePtr& fragment_shader_state,
	const FragmentOutputStatePtr& fragment_output_state
)
{
	COMMAND;

	if (vertex_input_state == m_current_vertex_input_state &&
	    pre_rasterization_shader_state == m_current_pre_rasterization_shader_state &&
	    fragment_shader_state == m_current_fragment_shader_state &&
	    fragment_output_state == m_current_fragment_output_state)
	{
		return;
	}

	m_current_vertex_input_state = vertex_input_state;
	m_current_pre_rasterization_shader_state = pre_rasterization_shader_state;
	m_current_fragment_shader_state = fragment_shader_state;
	m_current_fragment_output_state = fragment_output_state;

	vk::Pipeline pipeline = m_device_session->getGraphicsPipeline(
		vertex_input_state,
		pre_rasterization_shader_state,
		fragment_shader_state,
		fragment_output_state
	);

	{
		VULKAN_CALL(vkCmdBindPipeline);
		m_cmd_buf.bindPipeline(
			vk::PipelineBindPoint::eGraphics,
			pipeline,
			*m_dispatcher
		);
	}

	m_rec->addReferencedObject(vertex_input_state);
	m_rec->addReferencedObject(pre_rasterization_shader_state);
	m_rec->addReferencedObject(fragment_shader_state);
	m_rec->addReferencedObject(fragment_output_state);
}

void cgpu::GraphicsPassContext::bindIndexBuffer(
	const BufferPtr& buffer,
	vk::IndexType index_type,
	std::optional<Range<vk::DeviceSize>> range
)
{
	COMMAND;

	m_rec->addCmdResource(buffer, {vk::PipelineStageFlagBits2::eIndexInput, vk::AccessFlagBits2::eIndexRead});

	vk::BindIndexBuffer3InfoKHR info;
	info.addressRange.address = buffer->getDevicePtr() + (range ? range->offset : 0);
	info.addressRange.size = range ? range->size : buffer->getDesc().size;
	info.addressFlags = vk::AddressCommandFlagBitsKHR::eFullyBound;
	info.indexType = index_type;

	{
		VULKAN_CALL(vkCmdBindIndexBuffer3KHR);
		m_cmd_buf.bindIndexBuffer3KHR(info, *m_dispatcher);
	}
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
	COMMAND;

	pushParameters(data, size, alignment);

	{
		VULKAN_CALL(vkCmdDraw);
		m_cmd_buf.draw(
			vertex_count,
			instance_count,
			first_vertex,
			first_instance,
			*m_dispatcher
		);
	}
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
	COMMAND;

	pushParameters(data, size, alignment);

	{
		VULKAN_CALL(vkCmdDrawIndexed);
		m_cmd_buf.drawIndexed(
			index_count,
			instance_count,
			first_index,
			vertex_offset,
			first_instance,
			*m_dispatcher
		);
	}
}

void cgpu::GraphicsPassContext::setViewport(
	const vk::Viewport& viewport
)
{
	COMMAND;

	{
		VULKAN_CALL(vkCmdSetViewport);
		m_cmd_buf.setViewport(
			0,
			viewport,
			*m_dispatcher
		);
	}
}

void cgpu::GraphicsPassContext::setScissor(
	const vk::Rect2D& scissor
)
{
	COMMAND;

	{
		VULKAN_CALL(vkCmdSetScissor);
		m_cmd_buf.setScissor(
			0,
			scissor,
			*m_dispatcher
		);
	}
}

cgpu::GraphicsPassContext::GraphicsPassContext(CommandRecorder& rec, const DeviceSessionPtr& device_session, vk::CommandBuffer cmd_buf):
	PassContext{rec},
	m_device_session{device_session},
	m_dispatcher{&device_session->getDispatcher()},
	m_cmd_buf{cmd_buf}
{
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

void cgpu::GraphicsPassContext::pushParameters(const void* data, size_t size, size_t alignment)
{
	COMMAND;

	vk::DeviceAddress gpu_ptr = m_rec->writeParameters(data, size, alignment);

	vk::PushDataInfoEXT info;
	info.offset = 0;
	info.data.address = &gpu_ptr;
	info.data.size = sizeof(vk::DeviceAddress);

	{
		VULKAN_CALL(vkCmdPushDataEXT);
		m_cmd_buf.pushDataEXT(
			info,
			*m_dispatcher
		);
	}
}
