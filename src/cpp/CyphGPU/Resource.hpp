#pragma once

#include <CyphGPU/Queue.hpp>

#include <optional>

namespace cgpu
{
class Resource
{
public:
	virtual ~Resource() = default;

	Resource(const Resource&) = delete;
	Resource(Resource&&) = delete;

	Resource& operator=(const Resource&) = delete;
	Resource& operator=(Resource&&) = delete;

	[[nodiscard]]
	const std::flat_map<vk::Semaphore, uint64_t>& getReadSignals() const;

	[[nodiscard]]
	const std::optional<Queue::Signal>& tryGetReadWriteSignal() const;

	void addReadSignal(const Queue::Signal& signal);

	void setReadWriteSignal(const Queue::Signal& signal);

	void lock();

	void unlock();

protected:
	Resource() = default;

private:
	std::flat_map<vk::Semaphore, uint64_t> m_read_signals{};
	std::optional<Queue::Signal> m_read_write_signal{};

	std::mutex m_mutex{};
};
}
