#pragma once

#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/CommandRecorder.hpp>
#include <CyphGPU/Image.hpp>
#include <CyphGPU/ShaderTypes.hpp>

#include <glm/glm.hpp>

namespace cgpu
{
class CommandRecorder;

class PassContext
{
public:
	// ----- Commands -----

	void pushParameters(
		uint32_t slot,
		const void* data,
		size_t size,
		size_t alignment
	);

	template<class T>
	void pushParameters(
		uint32_t slot,
		const T& data
	)
	{
		pushParameters(slot, &data, sizeof(T), alignof(T));
	}

	[[nodiscard]]
	SampledImageHandle getSampledImageDescriptor(const ImagePtr& image, const Image::SampledDescriptorOverrides& overrides = {});

	[[nodiscard]]
	StorageImageHandle getStorageImageDescriptor(const ImagePtr& image, CommandRecorder::ResourceAccess access, const Image::StorageDescriptorOverrides& overrides = {});

	template<class T>
	[[nodiscard]]
	T* getBufferDevicePtr(const BufferPtr& buffer, CommandRecorder::ResourceAccess access, vk::DeviceSize offset = 0)
	{
		registerIndirectAccess(buffer, access);
		return buffer->getDevicePtrIndirect<T>(offset);
	}

	[[nodiscard]]
	UniformTexelBufferHandle getUniformTexelBufferDescriptor(const BufferPtr& buffer, vk::Format format, const Buffer::UniformTexelDescriptorOverrides& overrides = {});

	[[nodiscard]]
	StorageTexelBufferHandle getStorageTexelBufferDescriptor(const BufferPtr& buffer, vk::Format format, CommandRecorder::ResourceAccess access, const Buffer::StorageTexelDescriptorOverrides& overrides = {});

	void registerIndirectAccess(const ImagePtr& image, CommandRecorder::ResourceAccess access);
	void registerIndirectAccess(const BufferPtr& buffer, CommandRecorder::ResourceAccess access);

protected:
	CommandRecorder* m_rec;

	explicit PassContext(CommandRecorder& rec);
};
}
