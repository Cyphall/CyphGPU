#include "PassContext.hpp"

#include <CyphGPU/CommandRecorder.hpp>

void cgpu::PassContext::pushParameters(uint32_t slot, const void* data, size_t size, size_t alignment)
{
	m_rec->pushParameters(
		slot,
		data,
		size,
		alignment
	);
}

cgpu::SampledImageHandle cgpu::PassContext::getSampledImageDescriptor(const ImagePtr& image, const Image::SampledDescriptorOverrides& overrides)
{
	registerIndirectAccess(image, CommandRecorder::ResourceAccess::eReadonly);
	return image->getSampledDescriptorIndirect(overrides);
}

cgpu::StorageImageHandle cgpu::PassContext::getStorageImageDescriptor(const ImagePtr& image, CommandRecorder::ResourceAccess access, const Image::StorageDescriptorOverrides& overrides)
{
	registerIndirectAccess(image, access);
	return image->getStorageDescriptorIndirect(overrides);
}

cgpu::UniformTexelBufferHandle cgpu::PassContext::getUniformTexelBufferDescriptor(const BufferPtr& buffer, vk::Format format, const Buffer::UniformTexelDescriptorOverrides& overrides)
{
	registerIndirectAccess(buffer, CommandRecorder::ResourceAccess::eReadonly);
	return buffer->getUniformTexelDescriptorIndirect(format, overrides);
}

cgpu::StorageTexelBufferHandle cgpu::PassContext::getStorageTexelBufferDescriptor(const BufferPtr& buffer, vk::Format format, CommandRecorder::ResourceAccess access, const Buffer::StorageTexelDescriptorOverrides& overrides)
{
	registerIndirectAccess(buffer, access);
	return buffer->getStorageTexelDescriptorIndirect(format, overrides);
}

void cgpu::PassContext::registerIndirectAccess(const ImagePtr& image, CommandRecorder::ResourceAccess access)
{
	m_rec->addReferencedObject(image, access);
}

void cgpu::PassContext::registerIndirectAccess(const BufferPtr& buffer, CommandRecorder::ResourceAccess access)
{
	m_rec->addReferencedObject(buffer, access);
}

cgpu::PassContext::PassContext(CommandRecorder& rec):
	m_rec(&rec)
{
}