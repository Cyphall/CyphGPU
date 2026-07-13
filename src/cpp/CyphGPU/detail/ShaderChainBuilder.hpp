#pragma once

#include <CyphGPU/fwd.hpp>

#include <list>
#include <variant>
#include <vulkan/vulkan.hpp>

namespace cgpu::detail
{
class ShaderChainBuilder
{
public:
	explicit ShaderChainBuilder(DeviceSession& device_session);

	void addShader(
		const std::variant<std::vector<uint32_t>, std::string>& source,
		const char* entry_point,
		vk::ShaderStageFlagBits stage
	);

	[[nodiscard]]
	std::vector<vk::PipelineShaderStageCreateInfo> finalize() const;

private:
	using ShaderChain = vk::StructureChain<
		vk::PipelineShaderStageCreateInfo,
		vk::ShaderModuleCreateInfo,
		vk::ShaderDescriptorSetAndBindingMappingInfoEXT>;

	DeviceSession* m_device_session;

	std::vector<ShaderChain> m_shaders;
};
}
