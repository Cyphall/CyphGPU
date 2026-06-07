#include "Surface.hpp"

#include <CyphGPU/ContextSession/ContextSession.hpp>

cgpu::SurfacePtr cgpu::Surface::create(const ContextSessionPtr& context_session, Desc&& desc)
{
	return std::make_shared<cgpu::Surface>(PrivateKey{}, context_session, std::move(desc));
}

cgpu::Surface::Surface(PrivateKey, const ContextSessionPtr& context_session, Desc&& desc):
	m_context_session{context_session},
	m_desc{std::move(desc)},
	m_handle{m_desc.surface}
{}

cgpu::Surface::~Surface()
{
	if (m_desc.owned)
	{
		m_context_session->getHandle().destroySurfaceKHR(m_handle, nullptr, m_context_session->getDispatcher());
	}
}

const cgpu::ContextSessionPtr& cgpu::Surface::getContextSession() const
{
	return m_context_session;
}

const cgpu::Surface::Desc& cgpu::Surface::getDesc() const
{
	return m_desc;
}

const vk::SurfaceKHR& cgpu::Surface::getHandle() const
{
	return m_handle;
}
