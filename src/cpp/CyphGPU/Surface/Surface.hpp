#pragma once

#include <CyphGPU/fwd.hpp>

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
	static SurfaceRef create(const ContextSessionRef& context_session, Desc&& desc);

	explicit Surface(PrivateKey, const ContextSessionRef& context_session, Desc&& desc);

	Surface(const Surface&) = delete;
	Surface(Surface&&) = delete;

	Surface& operator=(const Surface&) = delete;
	Surface& operator=(Surface&&) = delete;

	~Surface();

	[[nodiscard]]
	const ContextSessionRef& getContextSession() const;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::SurfaceKHR& getHandle() const;

private:
	ContextSessionRef m_context_session;

	Desc m_desc;

	vk::SurfaceKHR m_handle{};
};
}
