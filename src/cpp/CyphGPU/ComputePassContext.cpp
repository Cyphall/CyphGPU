#include "ComputePassContext.hpp"

#include <CyphGPU/CommandRecorder.hpp>
#include <CyphGPU/TLAS.hpp>

cgpu::SampledImageHandle cgpu::ComputePassContext::getSampledImageDescriptor(const ImagePtr& image, const Image::SampledDescriptorOverrides& overrides)
{
	registerSampledImageIndirectAccess(image);
	return image->getSampledDescriptorIndirect(overrides);
}

cgpu::StorageImageHandle cgpu::ComputePassContext::getStorageImageDescriptor(const ImagePtr& image, StorageAccess access, const Image::StorageDescriptorOverrides& overrides)
{
	registerStorageImageIndirectAccess(image, access);
	return image->getStorageDescriptorIndirect(overrides);
}

cgpu::UniformTexelBufferHandle cgpu::ComputePassContext::getUniformTexelBufferDescriptor(const BufferPtr& buffer, vk::Format format, const Buffer::UniformTexelDescriptorOverrides& overrides)
{
	registerSampledBufferIndirectAccess(buffer);
	return buffer->getUniformTexelDescriptorIndirect(format, overrides);
}

cgpu::StorageTexelBufferHandle cgpu::ComputePassContext::getStorageTexelBufferDescriptor(const BufferPtr& buffer, StorageAccess access, vk::Format format, const Buffer::StorageTexelDescriptorOverrides& overrides)
{
	registerStorageBufferIndirectAccess(buffer, access);
	return buffer->getStorageTexelDescriptorIndirect(format, overrides);
}

vk::DeviceAddress cgpu::ComputePassContext::getTLASDevicePtr(const TLASPtr& tlas)
{
	registerTLASIndirectAccess(tlas);
	return tlas->getDevicePtr();
}

void cgpu::ComputePassContext::registerSampledImageIndirectAccess(const ImagePtr& image)
{
	m_rec->addCmdResource(image, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderSampledRead);
}

void cgpu::ComputePassContext::registerStorageImageIndirectAccess(const ImagePtr& image, StorageAccess access)
{
	m_rec->addCmdResource(image, vk::PipelineStageFlagBits2::eComputeShader, PassContext::toVk(access));
}

void cgpu::ComputePassContext::registerSampledBufferIndirectAccess(const BufferPtr& buffer)
{
	m_rec->addCmdResource(buffer, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderSampledRead);
}

void cgpu::ComputePassContext::registerStorageBufferIndirectAccess(const BufferPtr& buffer, StorageAccess access)
{
	m_rec->addCmdResource(buffer, vk::PipelineStageFlagBits2::eComputeShader, PassContext::toVk(access));
}

void cgpu::ComputePassContext::registerTLASIndirectAccess(const TLASPtr& tlas)
{
	m_rec->addCmdResource(tlas->getBuffer(), vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eAccelerationStructureReadKHR);
}

void cgpu::ComputePassContext::dispatch(
	const ComputeShaderStatePtr& compute_shader_state,
	const glm::uvec3& group_count,
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

	m_dispatch_cmds.push_back({
		.compute_shader_state = compute_shader_state,
		.group_count = group_count,
		.params_gpu_ptr = gpu_ptr,
	});
}

cgpu::ComputePassContext::ComputePassContext(CommandRecorder& rec, detail::BumpMemoryResource& bump_memory):
	PassContext{rec},
	m_dispatch_cmds{detail::BumpAllocator{bump_memory}}
{
}

const cgpu::detail::BumpList<cgpu::ComputePassContext::DispatchCmd>& cgpu::ComputePassContext::getDispatchCmds() const
{
	return m_dispatch_cmds;
}
