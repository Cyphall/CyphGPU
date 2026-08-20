#include "TLAS.hpp"

#include <CyphGPU/Buffer.hpp>
#include <CyphGPU/DeviceSession.hpp>

vk::AccelerationStructureBuildSizesInfoKHR cgpu::TLAS::calcSizes(const DeviceSessionPtr& device_session, const ASInfo& info)
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

cgpu::TLASPtr cgpu::TLAS::create(const DeviceSessionPtr& device_session, Desc&& desc)
{
	return std::make_shared<TLAS>(PrivateKey{}, device_session, std::move(desc));
}

cgpu::TLAS::TLAS(PrivateKey, const DeviceSessionPtr& device_session, Desc&& desc):
	m_device_session{device_session},
	m_desc{std::move(desc)},
	m_buffer{m_desc.buffer.lock()}
{
	createTLAS();
}

cgpu::TLAS::~TLAS()
{
	m_device_session->getHandle().destroyAccelerationStructureKHR(m_handle, nullptr, m_device_session->getDispatcher());
}

cgpu::DeviceSessionPtr cgpu::TLAS::getDeviceSession() const
{
	return m_device_session->shared_from_this();
}

const cgpu::TLAS::Desc& cgpu::TLAS::getDesc() const
{
	return m_desc;
}

const vk::AccelerationStructureKHR& cgpu::TLAS::getHandle()
{
	return m_handle;
}

const cgpu::BufferPtr& cgpu::TLAS::getBuffer() const
{
	return m_buffer;
}

const vk::DeviceAddress& cgpu::TLAS::getDevicePtr() const
{
	return m_device_ptr;
}

void cgpu::TLAS::fillVkStructs(const ASInfo& as_info, VkStructs& vk_structs)
{
	vk_structs.primitive_count = as_info.instance_count;

	vk_structs.geometry_info.geometryType = vk::GeometryTypeKHR::eInstances;
	vk_structs.geometry_info.geometry.instances = vk::AccelerationStructureGeometryInstancesDataKHR{};
	vk_structs.geometry_info.geometry.instances.arrayOfPointers = vk::False;
	vk_structs.geometry_info.geometry.instances.data.deviceAddress = 0;
	vk_structs.geometry_info.flags = {};

	vk_structs.build_geometry_info.type = vk::AccelerationStructureTypeKHR::eTopLevel;
	vk_structs.build_geometry_info.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
	vk_structs.build_geometry_info.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
	vk_structs.build_geometry_info.srcAccelerationStructure = nullptr;
	vk_structs.build_geometry_info.dstAccelerationStructure = nullptr;
	vk_structs.build_geometry_info.geometryCount = 1;
	vk_structs.build_geometry_info.pGeometries = &vk_structs.geometry_info;
	vk_structs.build_geometry_info.ppGeometries = nullptr;
	vk_structs.build_geometry_info.scratchData.deviceAddress = 0;
}

void cgpu::TLAS::createTLAS()
{
	auto range = m_desc.buffer_range ? *m_desc.buffer_range : Range<vk::DeviceSize>{0, m_buffer->getDesc().size};

	assert(range.offset % 256 == 0);
	assert(range.size == m_desc.sizes.accelerationStructureSize);

	vk::AccelerationStructureCreateInfo2KHR as_info;
	as_info.createFlags = {};
	as_info.addressRange.address = m_buffer->getDevicePtr() + range.offset;
	as_info.addressRange.size = range.size;
	as_info.addressFlags = vk::AddressCommandFlagBitsKHR::eFullyBound;
	as_info.type = vk::AccelerationStructureTypeKHR::eTopLevel;

	m_handle = m_device_session->getHandle().createAccelerationStructure2KHR(as_info, nullptr, m_device_session->getDispatcher());

	m_device_session->getHandle().setDebugUtilsObjectNameEXT(m_handle, m_desc.name, m_device_session->getDispatcher());

	vk::AccelerationStructureDeviceAddressInfoKHR device_ptr_info;
	device_ptr_info.accelerationStructure = m_handle;

	m_device_ptr = m_device_session->getHandle().getAccelerationStructureAddressKHR(device_ptr_info, m_device_session->getDispatcher());
}
