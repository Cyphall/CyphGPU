#include "ShaderBundle.hpp"

#include <format>

cgpu::ShaderBundle::ShaderBundle(cmrc::embedded_filesystem filesystem):
	m_filesystem{filesystem}
{
}

std::optional<std::span<const std::byte>> cgpu::ShaderBundle::tryGetShaderBlob(std::string_view identifier) const
{
	std::string path = std::format("{}.spv", identifier);

	if (!m_filesystem.is_file(path))
	{
		return std::nullopt;
	}

	cmrc::file spirv_file = m_filesystem.open(path);
	return std::span{
		reinterpret_cast<const std::byte*>(spirv_file.begin()),
		spirv_file.size(),
	};
}
