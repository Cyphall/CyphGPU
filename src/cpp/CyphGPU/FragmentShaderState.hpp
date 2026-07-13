#pragma once

#include <CyphGPU/fwd.hpp>

#include <variant>
#include <vulkan/vulkan.hpp>

namespace cgpu
{
class FragmentShaderState final
{
	class PrivateKey
	{};

public:
	struct Desc
	{
		struct FragmentShader
		{
			// Required
			std::variant<std::vector<uint32_t>, std::string> source;

			// Optional
			std::string entry_point{"main"};

			bool operator==(const FragmentShader&) const = default;
		};

		struct DepthState
		{
			// Required
			vk::CompareOp test_pass_condition;
			bool write_enabled;

			bool operator==(const DepthState&) const = default;
		};

		struct StencilState
		{
			// Required
			vk::StencilOpState front;
			vk::StencilOpState back;

			bool operator==(const StencilState&) const = default;
		};

		// Optional
		std::optional<FragmentShader> fragment_shader{};
		std::optional<DepthState> depth_state{};
		std::optional<StencilState> stencil_state{};
		uint32_t view_mask{0};

		bool operator==(const Desc&) const = default;
	};

	[[nodiscard]]
	static FragmentShaderStatePtr create(const DeviceSessionPtr& device_session, Desc&& desc);

	explicit FragmentShaderState(PrivateKey, DeviceSession& device_session, Desc&& desc);

	FragmentShaderState(const FragmentShaderState&) = delete;
	FragmentShaderState(FragmentShaderState&&) = delete;

	FragmentShaderState& operator=(const FragmentShaderState&) = delete;
	FragmentShaderState& operator=(FragmentShaderState&&) = delete;

	~FragmentShaderState();

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
struct std::hash<cgpu::FragmentShaderState::Desc>
{
	std::size_t operator()(const cgpu::FragmentShaderState::Desc& key) const noexcept;
};

template<>
struct std::hash<cgpu::FragmentShaderState::Desc::FragmentShader>
{
	std::size_t operator()(const cgpu::FragmentShaderState::Desc::FragmentShader& key) const noexcept;
};

template<>
struct std::hash<cgpu::FragmentShaderState::Desc::DepthState>
{
	std::size_t operator()(const cgpu::FragmentShaderState::Desc::DepthState& key) const noexcept;
};

template<>
struct std::hash<cgpu::FragmentShaderState::Desc::StencilState>
{
	std::size_t operator()(const cgpu::FragmentShaderState::Desc::StencilState& key) const noexcept;
};
