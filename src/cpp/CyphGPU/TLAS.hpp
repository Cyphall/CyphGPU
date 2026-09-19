#pragma once

#include <CyphGPU/fwd.hpp>
#include <CyphGPU/Utils.hpp>

#include <vulkan/vulkan.hpp>

namespace cgpu
{
class TLAS final
{
	class PrivateKey
	{};

public:
	struct ASInfo
	{
		/// Required.
		uint32_t instance_count CGPU_REQUIRED;
	};

	struct Desc
	{
		/// Required.
		std::string name CGPU_REQUIRED;
		/// Required.
		ASInfo as_info CGPU_REQUIRED;
		/// Required.
		std::weak_ptr<Buffer> buffer CGPU_REQUIRED;
		/// Required.
		vk::AccelerationStructureBuildSizesInfoKHR sizes CGPU_REQUIRED;
		/// Optional. Default: Whole range.
		///
		/// Final address must be aligned to 256 bytes.
		std::optional<Range<vk::DeviceSize>> buffer_range{};
	};

	[[nodiscard]]
	static vk::AccelerationStructureBuildSizesInfoKHR calcSizes(const DeviceSessionPtr& device_session, const ASInfo& info);

	[[nodiscard]]
	static TLASPtr create(const DeviceSessionPtr& device_session, Desc&& desc);

	explicit TLAS(PrivateKey, const DeviceSessionPtr& device_session, Desc&& desc);

	TLAS(const TLAS&) = delete;
	TLAS(TLAS&&) = delete;

	TLAS& operator=(const TLAS&) = delete;
	TLAS& operator=(TLAS&&) = delete;

	~TLAS();

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

	void createTLAS();
};
}
