#pragma once

#include <CyphGPU/Queue.hpp>

#include <flat_map>
#include <mutex>
#include <optional>

namespace cgpu
{
class Resource
{
public:
	enum class Type : uint8_t
	{
		eImage,
		eBuffer,
	};

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

	[[nodiscard]]
	Type getType() const;

protected:
	explicit Resource(Type type);

private:
	Type m_type;

	std::flat_map<vk::Semaphore, uint64_t> m_read_signals{};
	std::optional<Queue::Signal> m_read_write_signal{};

	std::mutex m_mutex{};
};
}
