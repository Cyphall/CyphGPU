#pragma once

#include <CyphGPU/fwd.hpp>

#include <variant>
#include <vulkan/vulkan.hpp>

namespace cgpu
{
class ComputeShaderState final
{
	class PrivateKey
	{};

public:
	struct Desc
	{
		struct ComputeShader
		{
			// Required
			std::variant<std::vector<uint32_t>, std::string> source;

			// Optional
			std::string entry_point{"main"};

			bool operator==(const ComputeShader&) const = default;
		};

		// Required
		ComputeShader compute_shader{};

		bool operator==(const Desc&) const = default;
	};

	[[nodiscard]]
	static ComputeShaderStatePtr create(const DeviceSessionPtr& device_session, Desc&& desc);

	explicit ComputeShaderState(PrivateKey, DeviceSession& device_session, Desc&& desc);

	ComputeShaderState(const ComputeShaderState&) = delete;
	ComputeShaderState(ComputeShaderState&&) = delete;

	ComputeShaderState& operator=(const ComputeShaderState&) = delete;
	ComputeShaderState& operator=(ComputeShaderState&&) = delete;

	~ComputeShaderState();

	[[nodiscard]]
	DeviceSessionPtr getDeviceSession() const;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::Pipeline& getHandle() const;

private:
	friend class DeviceSession;

	DeviceSession* m_device_session;

	Desc m_desc;

	vk::Pipeline m_handle;

	void createPipelineState();
};
}

template<>
struct std::hash<cgpu::ComputeShaderState::Desc>
{
	std::size_t operator()(const cgpu::ComputeShaderState::Desc& key) const noexcept;
};

template<>
struct std::hash<cgpu::ComputeShaderState::Desc::ComputeShader>
{
	std::size_t operator()(const cgpu::ComputeShaderState::Desc::ComputeShader& key) const noexcept;
};
