#pragma once

#include <CyphGPU/fwd.hpp>
#include <CyphGPU/Utils.hpp>

#include <vulkan/vulkan.hpp>

namespace cgpu
{
class BLAS final
{
	class PrivateKey
	{};

public:
	struct ASInfo
	{
		struct VertexBuffer
		{
			// Required
			uint32_t count;
			vk::Format format;

			// Optional
			std::optional<vk::DeviceSize> stride{};
		};

		struct IndexBuffer
		{
			// Required
			uint32_t count;
			vk::IndexType type;
		};

		// Required
		VertexBuffer vertex_buffer;

		// Optional
		std::optional<IndexBuffer> index_buffer{};
		bool opaque{false};
	};

	struct Desc
	{
		// Required
		std::string name;
		ASInfo as_info;
		std::weak_ptr<Buffer> buffer;
		vk::AccelerationStructureBuildSizesInfoKHR sizes;

		// Optional
		/// Final address must be aligned to 256 bytes.
		std::optional<Range<vk::DeviceSize>> buffer_range{};
	};

	[[nodiscard]]
	static vk::AccelerationStructureBuildSizesInfoKHR calcSizes(const DeviceSessionPtr& device_session, const ASInfo& info);

	[[nodiscard]]
	static BLASPtr create(const DeviceSessionPtr& device_session, Desc&& desc);

	explicit BLAS(PrivateKey, const DeviceSessionPtr& device_session, Desc&& desc);

	BLAS(const BLAS&) = delete;
	BLAS(BLAS&&) = delete;

	BLAS& operator=(const BLAS&) = delete;
	BLAS& operator=(BLAS&&) = delete;

	~BLAS();

	[[nodiscard]]
	DeviceSessionPtr getDeviceSession() const;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::AccelerationStructureKHR& getHandle();

	[[nodiscard]]
	const BufferPtr& getBuffer() const;

	[[nodiscard]]
	const vk::DeviceAddress& getDevicePtr() const;

private:
	friend class DeviceSession;
	friend class CommandRecorder;

	struct VkStructs
	{
		uint32_t primitive_count{};
		vk::AccelerationStructureGeometryKHR geometry_info{};
		vk::AccelerationStructureBuildGeometryInfoKHR build_geometry_info{};
	};

	DeviceSessionPtr m_device_session;

	Desc m_desc;

	BufferPtr m_buffer;

	vk::AccelerationStructureKHR m_handle;

	vk::DeviceAddress m_device_ptr{};

	static void fillVkStructs(const ASInfo& as_info, VkStructs& vk_structs);

	void createBLAS();
};
}
