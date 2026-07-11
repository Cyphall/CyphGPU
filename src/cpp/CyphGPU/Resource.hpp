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
	const std::optional<Queue::Signal>& tryGetSignal() const;

	void setSignal(const Queue::Signal& signal);

protected:

	Resource() = default;

private:
	std::optional<Queue::Signal> m_signal;
};
}
