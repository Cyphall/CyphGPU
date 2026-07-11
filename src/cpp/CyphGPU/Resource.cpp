#include "Resource.hpp"

const std::flat_map<vk::Semaphore, uint64_t>& cgpu::Resource::getReadSignals() const
{
	return m_read_signals;
}

const std::optional<cgpu::Queue::Signal>& cgpu::Resource::tryGetReadWriteSignal() const
{
	return m_read_write_signal;
}

void cgpu::Resource::addReadSignal(const Queue::Signal& signal)
{
	auto [it, inserted] = m_read_signals.try_emplace(signal.semaphore, signal.value);
	if (!inserted)
	{
		it->second = std::max(it->second, signal.value);
	}
}

void cgpu::Resource::setReadWriteSignal(const Queue::Signal& signal)
{
	m_read_write_signal = signal;

	m_read_signals.clear();
	m_read_signals.emplace(signal.semaphore, signal.value);
}

void cgpu::Resource::lock()
{
	m_mutex.lock();
}

void cgpu::Resource::unlock()
{
	m_mutex.unlock();
}
