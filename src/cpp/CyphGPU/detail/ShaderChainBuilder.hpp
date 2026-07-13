#pragma once

#include <vulkan/vulkan.hpp>

namespace cgpu::detail
{
class ShaderChainBuilder
{
public:
	explicit ShaderChainBuilder(std::span<const vk::DescriptorSetAndBindingMappingEXT> mappings);

	void addShader(
		std::span<const uint32_t> blob,
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

	std::span<const vk::DescriptorSetAndBindingMappingEXT> m_mappings;

	std::vector<ShaderChain> m_shaders;
};
}
