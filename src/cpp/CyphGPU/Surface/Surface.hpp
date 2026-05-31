#pragma once

#include <CyphGPU/DependencyObject.hpp>
#include <CyphGPU/ContextSession/ContextSession.hpp>

namespace cgpu
{
class Surface final: public DependencyObject<Surface>
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
	static SurfaceRef create(const ContextSessionRef& contextSession, Desc&& desc);

	explicit Surface(PrivateKey, const ContextSessionRef& contextSession, Desc&& desc);

	~Surface() override;

	[[nodiscard]]
	const Desc& getDesc() const;

	[[nodiscard]]
	const vk::SurfaceKHR& getHandle() const;

private:
	DependencyParent<ContextSession> m_contextSession;

	Desc m_desc;

	vk::SurfaceKHR m_surface{};
};
}
