#include "BLAS.hpp"

#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/DeviceSession.hpp>

#include <vulkan/vulkan_format_traits.hpp>

vk::AccelerationStructureBuildSizesInfoKHR cgpu::BLAS::calcSizes(const DeviceSessionPtr& device_session, const ASInfo& info)
{
	VkStructs vk_structs;
	fillVkStructs(info, vk_structs);

	return device_session->getHandle().getAccelerationStructureBuildSizesKHR(
		vk::AccelerationStructureBuildTypeKHR::eDevice,
		vk_structs.build_geometry_info,
		vk_structs.primitive_count,
		device_session->getDispatcher()
	);
}

cgpu::BLASPtr cgpu::BLAS::create(const DeviceSessionPtr& device_session, Desc&& desc)
{
	return std::make_shared<BLAS>(PrivateKey{}, device_session, std::move(desc));
}

cgpu::BLAS::BLAS(PrivateKey, const DeviceSessionPtr& device_session, Desc&& desc):
	m_device_session{device_session},
	m_desc{std::move(desc)},
	m_buffer{m_desc.buffer.lock()}
{
	createBLAS();
}

cgpu::BLAS::~BLAS()
{
	m_device_session->getHandle().destroyAccelerationStructureKHR(m_handle, nullptr, m_device_session->getDispatcher());
}

cgpu::DeviceSessionPtr cgpu::BLAS::getDeviceSession() const
{
	return m_device_session->shared_from_this();
}

const cgpu::BLAS::Desc& cgpu::BLAS::getDesc() const
{
	return m_desc;
}

const vk::AccelerationStructureKHR& cgpu::BLAS::getHandle()
{
	return m_handle;
}

const cgpu::BufferPtr& cgpu::BLAS::getBuffer() const
{
	return m_buffer;
}

const vk::DeviceAddress& cgpu::BLAS::getDevicePtr() const
{
	return m_device_ptr;
}

void cgpu::BLAS::fillVkStructs(const ASInfo& as_info, VkStructs& vk_structs)
{
	vk_structs.primitive_count = as_info.vertex_buffer.count / 3;

	vk_structs.geometry_info.geometryType = vk::GeometryTypeKHR::eTriangles;
	vk_structs.geometry_info.geometry.triangles = vk::AccelerationStructureGeometryTrianglesDataKHR{};
	vk_structs.geometry_info.geometry.triangles.vertexFormat = as_info.vertex_buffer.format;
	vk_structs.geometry_info.geometry.triangles.vertexData.deviceAddress = 0;
	vk_structs.geometry_info.geometry.triangles.vertexStride = as_info.vertex_buffer.stride ? *as_info.vertex_buffer.stride : vk::blockSize(as_info.vertex_buffer.format);
	vk_structs.geometry_info.geometry.triangles.maxVertex = as_info.vertex_buffer.count;
	vk_structs.geometry_info.geometry.triangles.indexType = as_info.index_buffer ? as_info.index_buffer->type : vk::IndexType::eNoneKHR;
	vk_structs.geometry_info.geometry.triangles.indexData.deviceAddress = 0;
	vk_structs.geometry_info.geometry.triangles.transformData.deviceAddress = 0;
	vk_structs.geometry_info.flags = as_info.opaque ? vk::GeometryFlagBitsKHR::eOpaque : vk::GeometryFlagsKHR{};

	vk_structs.build_geometry_info.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
	vk_structs.build_geometry_info.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
	vk_structs.build_geometry_info.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
	vk_structs.build_geometry_info.srcAccelerationStructure = nullptr;
	vk_structs.build_geometry_info.dstAccelerationStructure = nullptr;
	vk_structs.build_geometry_info.geometryCount = 1;
	vk_structs.build_geometry_info.pGeometries = &vk_structs.geometry_info;
	vk_structs.build_geometry_info.ppGeometries = nullptr;
	vk_structs.build_geometry_info.scratchData.deviceAddress = 0;
}

void cgpu::BLAS::createBLAS()
{
	auto range = m_desc.buffer_range ? *m_desc.buffer_range : Range<vk::DeviceSize>{0, m_buffer->getDesc().size};

	assert(range.offset % 256 == 0);
	assert(range.size == m_desc.sizes.accelerationStructureSize);

	vk::AccelerationStructureCreateInfoKHR as_info;
	as_info.createFlags = {};
	as_info.buffer = m_buffer->getHandle();
	as_info.offset = range.offset;
	as_info.size = range.size;
	as_info.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
	as_info.deviceAddress = 0;

	m_handle = m_device_session->getHandle().createAccelerationStructureKHR(as_info, nullptr, m_device_session->getDispatcher());

	m_device_session->getHandle().setDebugUtilsObjectNameEXT(m_handle, m_desc.name, m_device_session->getDispatcher());

	vk::AccelerationStructureDeviceAddressInfoKHR device_ptr_info;
	device_ptr_info.accelerationStructure = m_handle;

	m_device_ptr = m_device_session->getHandle().getAccelerationStructureAddressKHR(device_ptr_info, m_device_session->getDispatcher());
}
