#include "Resource.hpp"

const std::flat_map<vk::Semaphore, uint64_t>& cgpu::detail::Resource::getReadSignals() const
{
	return m_read_signals;
}

const std::optional<cgpu::Queue::Signal>& cgpu::detail::Resource::tryGetReadWriteSignal() const
{
	return m_read_write_signal;
}

void cgpu::detail::Resource::addReadSignal(const Queue::Signal& signal)
{
	auto [it, inserted] = m_read_signals.try_emplace(signal.semaphore, signal.value);
	if (!inserted)
	{
		it->second = std::max(it->second, signal.value);
	}
}

void cgpu::detail::Resource::setReadWriteSignal(const Queue::Signal& signal)
{
	m_read_write_signal = signal;

	m_read_signals.clear();
	m_read_signals.emplace(signal.semaphore, signal.value);
}

void cgpu::detail::Resource::clearSignals()
{
	m_read_write_signal = std::nullopt;
	m_read_signals.clear();
}

void cgpu::detail::Resource::lock()
{
	m_mutex.lock();
}

void cgpu::detail::Resource::unlock()
{
	m_mutex.unlock();
}

cgpu::detail::Resource::Type cgpu::detail::Resource::getType() const
{
	return m_type;
}

cgpu::detail::Resource::Resource(Type type):
	m_type{type}
{
}
