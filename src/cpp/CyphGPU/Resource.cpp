#include "Resource.hpp"

const std::optional<cgpu::Queue::Signal>& cgpu::Resource::tryGetSignal() const
{
	return m_signal;
}

void cgpu::Resource::setSignal(const Queue::Signal& signal)
{
	m_signal = signal;
}
