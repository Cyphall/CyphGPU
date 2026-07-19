#include "ShaderBundle.hpp"

#include <format>

cgpu::ShaderBundle::ShaderBundle(cmrc::embedded_filesystem filesystem):
	m_filesystem{filesystem}
{
}

std::optional<std::span<const std::byte>> cgpu::ShaderBundle::tryGetShaderBlob(std::string_view identifier) const
{
	try
	{
		cmrc::file spirv_file = m_filesystem.open(std::format("{}.spv", identifier));
		return std::span{
			reinterpret_cast<const std::byte*>(spirv_file.begin()),
			spirv_file.size(),
		};
	}
	catch (...)
	{
		return std::nullopt;
	}
}
