#pragma once

#include <cmrc/cmrc.hpp>
#include <optional>
#include <span>

namespace cgpu
{
class ShaderBundle
{
public:
	explicit ShaderBundle(cmrc::embedded_filesystem filesystem);

	[[nodiscard]]
	std::optional<std::span<const std::byte>> tryGetShaderBlob(std::string_view identifier) const;

private:
	cmrc::embedded_filesystem m_filesystem;
};
}

#define CGPU_DECLARE_SHADER_BUNDLE(name) \
	CMRC_DECLARE(name);                  \
	static cgpu::ShaderBundle name{cmrc::name::get_filesystem()};
