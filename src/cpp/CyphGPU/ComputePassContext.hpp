#pragma once

#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/detail/BumpAllocator.hpp>
#include <CyphGPU/fwd.hpp>
#include <CyphGPU/Image.hpp>
#include <CyphGPU/PassContext.hpp>

#include <glm/glm.hpp>

namespace cgpu
{
class ComputePassContext final : public PassContext
{
public:
	[[nodiscard]]
	SampledImageHandle getSampledImageDescriptor(
		const ImagePtr& image
	);

	[[nodiscard]]
	SampledImageHandle getSampledImageDescriptor(
		const ImagePtr& image,
		const Image::SampledDescriptorOverrides& overrides
	);

	[[nodiscard]]
	StorageImageHandle getStorageImageDescriptor(
		const ImagePtr& image,
		StorageAccess access
	);

	[[nodiscard]]
	StorageImageHandle getStorageImageDescriptor(
		const ImagePtr& image,
		StorageAccess access,
		const Image::StorageDescriptorOverrides& overrides
	);

	template<class T>
	[[nodiscard]]
	T* getBufferDevicePtr(
		const BufferPtr& buffer,
		StorageAccess access,
		vk::DeviceSize offset = 0
	)
	{
		registerStorageBufferIndirectAccess(buffer, access);
		return buffer->getDevicePtrIndirect<T>(offset);
	}

	[[nodiscard]]
	UniformTexelBufferHandle getUniformTexelBufferDescriptor(
		const BufferPtr& buffer,
		vk::Format format
	);

	[[nodiscard]]
	UniformTexelBufferHandle getUniformTexelBufferDescriptor(
		const BufferPtr& buffer,
		vk::Format format,
		const Buffer::UniformTexelDescriptorOverrides& overrides
	);

	[[nodiscard]]
	StorageTexelBufferHandle getStorageTexelBufferDescriptor(
		const BufferPtr& buffer,
		StorageAccess access,
		vk::Format format
	);

	[[nodiscard]]
	StorageTexelBufferHandle getStorageTexelBufferDescriptor(
		const BufferPtr& buffer,
		StorageAccess access,
		vk::Format format,
		const Buffer::StorageTexelDescriptorOverrides& overrides
	);

	[[nodiscard]]
	vk::DeviceAddress getTLASDevicePtr(
		const TLASPtr& tlas
	);

	void registerSampledImageIndirectAccess(const ImagePtr& image);
	void registerStorageImageIndirectAccess(const ImagePtr& image, StorageAccess access);
	void registerSampledBufferIndirectAccess(const BufferPtr& buffer);
	void registerStorageBufferIndirectAccess(const BufferPtr& buffer, StorageAccess access);
	void registerTLASIndirectAccess(const TLASPtr& tlas);

	// ----- Commands -----

	void dispatch(
		const ComputeShaderStatePtr& compute_shader_state,
		const glm::uvec3& group_count,
		const void* data,
		size_t size,
		size_t alignment
	);

	template<class T>
	void dispatch(
		const ComputeShaderStatePtr& compute_shader_state,
		const glm::uvec3& group_count,
		const T& data
	)
	{
		dispatch(
			compute_shader_state,
			group_count,
			&data,
			sizeof(T),
			alignof(T)
		);
	}

	void dispatch(
		const ComputeShaderStatePtr& compute_shader_state,
		const glm::uvec3& thread_count,
		const glm::uvec3& group_size,
		const void* data,
		size_t size,
		size_t alignment
	);

	template<class T>
	void dispatch(
		const ComputeShaderStatePtr& compute_shader_state,
		const glm::uvec3& thread_count,
		const glm::uvec3& group_size,
		const T& data
	)
	{
		dispatch(
			compute_shader_state,
			thread_count,
			group_size,
			&data,
			sizeof(T),
			alignof(T)
		);
	}

private:
	friend class CommandRecorder;

	struct DispatchCmd
	{
		ComputeShaderStatePtr compute_shader_state;
		glm::uvec3 group_count;
		vk::DeviceAddress params_gpu_ptr;
	};

	detail::BumpList<DispatchCmd>* m_dispatch_cmds;

	explicit ComputePassContext(CommandRecorder& rec, detail::BumpList<DispatchCmd>& dispatch_cmds);
};
}
