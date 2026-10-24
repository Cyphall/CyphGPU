#pragma once

#include <CyphGPU/Queue.hpp>

#include <flat_map>
#include <optional>
#include <shared_mutex>

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

protected:
	friend class CommandRecorder;

	enum class Type : uint8_t
	{
		eImage,
		eBuffer,
	};

	explicit Resource(Type type);

	[[nodiscard]]
	const std::flat_map<vk::Semaphore, uint64_t>& getReadSignals() const;

	[[nodiscard]]
	const std::optional<Queue::Signal>& tryGetReadWriteSignal() const;

	void addReadSignal(const Queue::Signal& signal);

	void setReadWriteSignal(const Queue::Signal& signal);

	void clearSignals();

	void lock();

	void unlock();

	[[nodiscard]]
	Type getType() const;

private:
	Type m_type;

	std::flat_map<vk::Semaphore, uint64_t> m_read_signals{};
	std::optional<Queue::Signal> m_read_write_signal{};

	std::shared_mutex m_mutex{};
};
}
