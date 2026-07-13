#include "ShaderBundle.hpp"

#include <format>

cgpu::ShaderBundle::ShaderBundle(cmrc::embedded_filesystem filesystem):
	m_filesystem{filesystem}
{
}

std::optional<std::span<const uint32_t>> cgpu::ShaderBundle::tryGetShaderBlob(std::string_view identifier) const
{
	cmrc::file spirvFile{};
	try
	{
		spirvFile = m_filesystem.open(std::format("{}.spv", identifier));
	}
	catch (...)
	{
		return std::nullopt;
	}

	if (spirvFile.size() % 4 != 0)
	{
		throw std::runtime_error(std::format("Shader file size for identifier \"{}\" is not a multiple of 4.", identifier));
	}

	return std::span{
		reinterpret_cast<const uint32_t*>(spirvFile.begin()),
		spirvFile.size() / 4,
	};
}
