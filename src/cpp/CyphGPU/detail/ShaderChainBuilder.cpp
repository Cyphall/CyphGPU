#include "ShaderChainBuilder.hpp"

#include <CyphGPU/Context.hpp>
#include <CyphGPU/Device.hpp>
#include <CyphGPU/DeviceSession.hpp>

namespace
{
template<class... Ts>
struct overloaded : Ts...
{
	using Ts::operator()...;
};
}

cgpu::detail::ShaderChainBuilder::ShaderChainBuilder(DeviceSession& device_session):
	m_device_session{&device_session}
{
}

void cgpu::detail::ShaderChainBuilder::addShader(
	const std::variant<std::vector<uint32_t>, std::string>& source,
	const char* entry_point,
	vk::ShaderStageFlagBits stage
)
{
	ShaderChain& chain = m_shaders.emplace_back();

	auto& stage_info = chain.get<vk::PipelineShaderStageCreateInfo>();
	stage_info.flags = {};
	stage_info.stage = stage;
	stage_info.module = nullptr;
	stage_info.pName = entry_point;
	stage_info.pSpecializationInfo = nullptr;

	std::span<const uint32_t> blob = std::visit(
		overloaded{
			[&](const std::vector<uint32_t>& value) -> std::span<const uint32_t> {
				return value;
			},
			[&](const std::string& value) -> std::span<const uint32_t> {
				return m_identifier_data_storage.emplace_back(
					m_device_session->getDevice()->getContextSession()->getContext()->getDesc().shader_identifier_resolver.value()(value)
				);
			},
		},
		source
	);

	auto& module_info = chain.get<vk::ShaderModuleCreateInfo>();
	module_info.flags = {};
	module_info.codeSize = static_cast<uint32_t>(blob.size() * sizeof(uint32_t));
	module_info.pCode = blob.data();

	auto& mapping_info = chain.get<vk::ShaderDescriptorSetAndBindingMappingInfoEXT>();
	mapping_info.mappingCount = static_cast<uint32_t>(m_device_session->getMappings().size());
	mapping_info.pMappings = m_device_session->getMappings().data();
}

std::vector<vk::PipelineShaderStageCreateInfo> cgpu::detail::ShaderChainBuilder::finalize() const
{
	std::vector<vk::PipelineShaderStageCreateInfo> packed_shader_infos;
	for (const auto& chain : m_shaders)
	{
		packed_shader_infos.emplace_back(chain.get());
	}

	return packed_shader_infos;
}
