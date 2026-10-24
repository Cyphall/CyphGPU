#include "Resource.hpp"

void cgpu::Resource::makeImmutable()
{
	lock();

	assert(!m_immutable.load() && "This resource is already immutable.");
	assert(m_layout_initialized && "Images must have their layout initialized before being made immutable.");

	m_read_signals.clear();

	m_immutable.store(true, std::memory_order_release);

	unlock();
}

bool cgpu::Resource::isImmutable() const
{
	return m_immutable.load();
}

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

void cgpu::Resource::clearSignals()
{
	m_read_write_signal = std::nullopt;
	m_read_signals.clear();
}

void cgpu::Resource::lock()
{
	m_mutex.lock();
}

void cgpu::Resource::unlock()
{
	m_mutex.unlock();
}

cgpu::Resource::Type cgpu::Resource::getType() const
{
	return m_type;
}

bool cgpu::Resource::isLayoutInitialized() const
{
	return m_layout_initialized;
}

void cgpu::Resource::setLayoutInitialized(bool init)
{
	m_layout_initialized = init;
}

cgpu::Resource::Resource(Type type):
	m_type{type}
{
}
