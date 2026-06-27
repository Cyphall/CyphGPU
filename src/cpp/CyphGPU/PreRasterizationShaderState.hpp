#pragma once

#include <CyphGPU/fwd.hpp>

#include <vulkan/vulkan.hpp>

namespace cgpu
{
class PreRasterizationShaderState final
{
	class PrivateKey
	{};

public:
	struct Desc
	{
		struct VertexShader
		{
			// Required
			std::vector<uint32_t> blob;

			// Optional
			std::string entry_point{"main"};

			bool operator==(const VertexShader&) const = default;
		};

		struct GeometryShader
		{
			// Required
			std::vector<uint32_t> blob;

			// Optional
			std::string entry_point{"main"};

			bool operator==(const GeometryShader&) const = default;
		};

		// Required
		VertexShader vertex_shader;

		// Optional
		std::optional<GeometryShader> geometry_shader{};
		bool depth_clamp{false};
		vk::PolygonMode polygon_mode{vk::PolygonMode::eFill};
		vk::CullModeFlags cull_mode{vk::CullModeFlagBits::eBack};
		vk::FrontFace front_face{vk::FrontFace::eCounterClockwise};
		float line_width{1.0f};
		uint32_t view_mask{0};

		bool operator==(const Desc&) const = default;
	};

	[[nodiscard]]
	static PreRasterizationShaderStatePtr create(const DeviceSessionPtr& device_session, Desc&& desc);

	explicit PreRasterizationShaderState(PrivateKey, DeviceSession& device_session, Desc&& desc);

	PreRasterizationShaderState(const PreRasterizationShaderState&) = delete;
	PreRasterizationShaderState(PreRasterizationShaderState&&) = delete;

	PreRasterizationShaderState& operator=(const PreRasterizationShaderState&) = delete;
	PreRasterizationShaderState& operator=(PreRasterizationShaderState&&) = delete;

	~PreRasterizationShaderState();

	[[nodiscard]]
	DeviceSessionPtr getDeviceSession() const;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::Pipeline& getHandle();

private:
	friend class DeviceSession;

	DeviceSession* m_device_session;

	Desc m_desc;

	vk::Pipeline m_handle;

	void createPipelineState();
};
}

template<>
struct std::hash<cgpu::PreRasterizationShaderState::Desc>
{
	std::size_t operator()(const cgpu::PreRasterizationShaderState::Desc& key) const noexcept;
};

template<>
struct std::hash<cgpu::PreRasterizationShaderState::Desc::VertexShader>
{
	std::size_t operator()(const cgpu::PreRasterizationShaderState::Desc::VertexShader& key) const noexcept;
};

template<>
struct std::hash<cgpu::PreRasterizationShaderState::Desc::GeometryShader>
{
	std::size_t operator()(const cgpu::PreRasterizationShaderState::Desc::GeometryShader& key) const noexcept;
};
