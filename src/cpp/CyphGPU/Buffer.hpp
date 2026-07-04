#pragma once

#include <CyphGPU/fwd.hpp>
#include <CyphGPU/MemoryType.hpp>
#include <CyphGPU/Queue.hpp>
#include <CyphGPU/Utils.hpp>

#include <flat_map>
#include <glm/glm.hpp>
#include <optional>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace cgpu
{
class Buffer final
{
	class PrivateKey
	{};

public:
	struct Desc
	{
		struct ExistingHandle
		{
			vk::Buffer buffer;
			/// Must be valid for host-visible memory
			std::optional<std::byte*> host_ptr;
		};

		// Required
		std::string name;
		vk::DeviceSize size;
		vk::BufferUsageFlags2 usages;

		// Optional
		MemoryType memory_type{MemoryType::eGPUHighPrio};
		vk::DeviceSize min_alignment{1};
		/// Other desc parameters must still match what the buffer was created with.
		std::optional<ExistingHandle> existing_handle{};
	};

	struct UniformTexelDescriptorOverrides
	{
		/// Default: Whole range.
		std::optional<Range<vk::DeviceSize>> byte_range;
	};

	struct StorageTexelDescriptorOverrides
	{
		/// Default: Whole range.
		std::optional<Range<vk::DeviceSize>> byte_range;
	};

	[[nodiscard]]
	static BufferPtr create(const DeviceSessionPtr& device_session, Desc&& desc);

	explicit Buffer(PrivateKey, const DeviceSessionPtr& device_session, Desc&& desc);

	Buffer(const Buffer&) = delete;
	Buffer(Buffer&&) = delete;

	Buffer& operator=(const Buffer&) = delete;
	Buffer& operator=(Buffer&&) = delete;

	~Buffer();

	[[nodiscard]]
	const DeviceSessionPtr& getDeviceSession() const;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::Buffer& getHandle();

	[[nodiscard]]
	const vk::DeviceAddress& getDevicePtr();

	[[nodiscard]]
	std::byte* getHostPtr();

	[[nodiscard]]
	const std::byte* getHostPtr() const;

	template<class T = std::byte>
	[[nodiscard]]
	T* getHostPtr(vk::DeviceSize offset = 0)
	{
		return reinterpret_cast<T*>(getHostPtr() + offset);
	}

	template<class T = std::byte>
	[[nodiscard]]
	const T* getHostPtr(vk::DeviceSize offset = 0) const
	{
		return reinterpret_cast<const T*>(getHostPtr() + offset);
	}

	[[nodiscard]]
	uint32_t getUniformTexelDescriptor(vk::Format format, const UniformTexelDescriptorOverrides& overrides = {});

	[[nodiscard]]
	uint32_t getStorageTexelDescriptor(vk::Format format, const StorageTexelDescriptorOverrides& overrides = {});

	[[nodiscard]]
	const std::optional<Queue::SubmitSync>& tryGetSubmitSync() const;

	void setSubmitSync(const Queue::SubmitSync& submit_sync);

private:
	struct UniformTexelDescriptorInfo
	{
		vk::Format format{};
		Range<vk::DeviceSize> byte_range{};

		auto operator<=>(const UniformTexelDescriptorInfo&) const = default;
	};

	struct StorageTexelDescriptorInfo
	{
		vk::Format format{};
		Range<vk::DeviceSize> byte_range{};

		auto operator<=>(const StorageTexelDescriptorInfo&) const = default;
	};

	DeviceSessionPtr m_device_session;

	Desc m_desc;

	vk::Buffer m_handle{};
	std::optional<VmaAllocation> m_alloc{};

	vk::DeviceAddress m_device_ptr{};
	std::optional<std::byte*> m_host_ptr{};

	std::flat_map<UniformTexelDescriptorInfo, uint32_t> m_uniform_texel_cache;
	std::flat_map<StorageTexelDescriptorInfo, uint32_t> m_storage_texel_cache;

	std::optional<Queue::SubmitSync> m_submit_sync;

	void createBuffer();
};
}
