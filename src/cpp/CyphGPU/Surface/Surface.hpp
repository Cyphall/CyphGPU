#pragma once

#include <CyphGPU/ContextSession/ContextSession.hpp>
#include <CyphGPU/DependencyObject.hpp>

namespace cgpu
{
class Surface final : public DependencyObject<Surface>
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

	~Surface() override;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::SurfaceKHR& getHandle() const;

private:
	DependencyParent<ContextSession> m_context_session;

	Desc m_desc;

	vk::SurfaceKHR m_surface{};
};
}
