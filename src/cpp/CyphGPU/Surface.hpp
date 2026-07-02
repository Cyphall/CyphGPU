#pragma once

#include <CyphGPU/fwd.hpp>

#include <vulkan/vulkan.hpp>

namespace cgpu
{
class Surface final
{
	class PrivateKey
	{};

public:
	struct Desc
	{
		// Required
		vk::SurfaceKHR surface;

		// Optional
		bool owned{true};
	};

	[[nodiscard]]
	static SurfacePtr create(const ContextSessionPtr& context_session, Desc&& desc);

	explicit Surface(PrivateKey, const ContextSessionPtr& context_session, Desc&& desc);

	Surface(const Surface&) = delete;
	Surface(Surface&&) = delete;

	Surface& operator=(const Surface&) = delete;
	Surface& operator=(Surface&&) = delete;

	~Surface();

	[[nodiscard]]
	const ContextSessionPtr& getContextSession() const;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::SurfaceKHR& getHandle();

private:
	ContextSessionPtr m_context_session;

	Desc m_desc;

	vk::SurfaceKHR m_handle{};
};
}
