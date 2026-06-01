#include "Surface.hpp"

cgpu::SurfaceRef cgpu::Surface::create(const ContextSessionRef& contextSession, Desc&& desc)
{
	cgpu::SurfaceRef ref = std::make_shared<cgpu::Surface>(PrivateKey{}, contextSession, std::move(desc));
	ref->m_weakThis = ref;
	return ref;
}

cgpu::Surface::Surface(PrivateKey, const ContextSessionRef& contextSession, Desc&& desc):
	m_contextSession{contextSession},
	m_desc{std::move(desc)},
	m_surface{m_desc.surface}
{}

cgpu::Surface::~Surface()
{
	if (m_desc.owned)
	{
		m_contextSession->getHandle().destroySurfaceKHR(m_surface, nullptr, m_contextSession->getDispatcher());
	}
}

const cgpu::Surface::Desc& cgpu::Surface::getDesc() const
{
	return m_desc;
}

const vk::SurfaceKHR& cgpu::Surface::getHandle() const
{
	return m_surface;
}
