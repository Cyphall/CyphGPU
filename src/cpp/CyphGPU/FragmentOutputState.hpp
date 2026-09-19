#pragma once

#include <CyphGPU/fwd.hpp>
#include <CyphGPU/Utils.hpp>

#include <boost/container/static_vector.hpp>
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

namespace cgpu
{
class FragmentOutputState final
{
	class PrivateKey
	{};

public:
	struct Desc
	{
		struct BlendComponentState
		{
			/// Required.
			vk::BlendFactor src_factor CGPU_REQUIRED;
			/// Required.
			vk::BlendFactor dst_factor CGPU_REQUIRED;
			/// Required.
			vk::BlendOp op CGPU_REQUIRED;

			bool operator==(const BlendComponentState&) const = default;
		};

		struct BlendState
		{
			/// Required.
			BlendComponentState color CGPU_REQUIRED;
			/// Required.
			BlendComponentState alpha CGPU_REQUIRED;

			bool operator==(const BlendState&) const = default;
		};

		struct ColorAttachment
		{
			/// Required.
			vk::Format format CGPU_REQUIRED;
			/// Optional. Default: No blending.
			std::optional<BlendState> blend{};
			/// Optional. Default: RGBA.
			vk::ColorComponentFlags write_mask{
				vk::ColorComponentFlagBits::eR |
				vk::ColorComponentFlagBits::eG |
				vk::ColorComponentFlagBits::eB |
				vk::ColorComponentFlagBits::eA
			};

			bool operator==(const ColorAttachment&) const = default;
		};

		struct DepthStencilAttachment
		{
			/// Required.
			vk::Format format CGPU_REQUIRED;
			/// Optional. Default: True if the format has a depth aspect.
			std::optional<bool> enable_depth{};
			/// Optional. Default: True if the format has a stencil aspect.
			std::optional<bool> enable_stencil{};

			bool operator==(const DepthStencilAttachment&) const = default;
		};

		/// Optional. Default: No color attachment.
		boost::container::static_vector<ColorAttachment, 8> color_attachments{};
		/// Optional. Default: No depth-stencil attachment.
		std::optional<DepthStencilAttachment> depth_stencil_attachment{};
		/// Optional. Default: 1.
		vk::SampleCountFlagBits samples{vk::SampleCountFlagBits::e1};
		/// Optional. Default: Transparent black.
		glm::vec4 blend_constants{0.0f, 0.0f, 0.0f, 0.0f};

		bool operator==(const Desc&) const = default;
	};

	[[nodiscard]]
	static FragmentOutputStatePtr create(const DeviceSessionPtr& device_session, Desc&& desc);

	explicit FragmentOutputState(PrivateKey, DeviceSession& device_session, Desc&& desc);

	FragmentOutputState(const FragmentOutputState&) = delete;
	FragmentOutputState(FragmentOutputState&&) = delete;

	FragmentOutputState& operator=(const FragmentOutputState&) = delete;
	FragmentOutputState& operator=(FragmentOutputState&&) = delete;

	~FragmentOutputState();

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
struct std::hash<cgpu::FragmentOutputState::Desc>
{
	std::size_t operator()(const cgpu::FragmentOutputState::Desc& key) const noexcept;
};

template<>
struct std::hash<cgpu::FragmentOutputState::Desc::BlendComponentState>
{
	std::size_t operator()(const cgpu::FragmentOutputState::Desc::BlendComponentState& key) const noexcept;
};

template<>
struct std::hash<cgpu::FragmentOutputState::Desc::BlendState>
{
	std::size_t operator()(const cgpu::FragmentOutputState::Desc::BlendState& key) const noexcept;
};

template<>
struct std::hash<cgpu::FragmentOutputState::Desc::ColorAttachment>
{
	std::size_t operator()(const cgpu::FragmentOutputState::Desc::ColorAttachment& key) const noexcept;
};

template<>
struct std::hash<cgpu::FragmentOutputState::Desc::DepthStencilAttachment>
{
	std::size_t operator()(const cgpu::FragmentOutputState::Desc::DepthStencilAttachment& key) const noexcept;
};
