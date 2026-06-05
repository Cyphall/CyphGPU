#pragma once

#include <concepts>
#include <memory>
#include <optional>

namespace cgpu
{
template<class TDerived>
class DependencyObject
{
public:
	virtual ~DependencyObject() = default;

	DependencyObject(const DependencyObject&) = delete;
	DependencyObject(DependencyObject&&) = delete;

	DependencyObject& operator=(const DependencyObject&) = delete;
	DependencyObject& operator=(DependencyObject&&) = delete;

	std::shared_ptr<TDerived> asRef()
	{
		return {m_owning_ptr.lock(), static_cast<TDerived*>(this)};
	}

	std::shared_ptr<const TDerived> asRef() const
	{
		return {m_owning_ptr.lock(), static_cast<const TDerived*>(this)};
	}

protected:
	template<class, class TParent>
	requires std::derived_from<TParent, DependencyObject<TParent>>
	friend class DependencyObjectChild;

	// Self-owned constructor
	explicit DependencyObject():
		m_owning_ptr{m_weak_this}
	{
	}

	// External-owned constructor
	explicit DependencyObject(std::weak_ptr<void>& external_owning_ptr):
		m_owning_ptr{external_owning_ptr}
	{
	}

	// Similar to weak_this in std::enable_shared_from_this
	// Will be empty for external-owned objects
	std::weak_ptr<void> m_weak_this;

	// For independent objects, points to our m_weak_this
	// For external-owned objects, points to the m_owning_ptr of the owning object
	std::weak_ptr<void>& m_owning_ptr;
};

template<class TParent>
requires std::derived_from<TParent, DependencyObject<TParent>>
class DependencyParent
{
public:
	explicit DependencyParent(const std::shared_ptr<TParent>& parent):
		m_parent{*parent},
		m_parent_ref{parent}
	{
	}

	explicit DependencyParent(TParent& parent):
		m_parent{parent}
	{
	}

	DependencyParent(const DependencyParent&) = delete;
	DependencyParent(DependencyParent&&) = delete;

	DependencyParent& operator=(const DependencyParent&) = delete;
	DependencyParent& operator=(DependencyParent&&) = delete;

	std::shared_ptr<TParent> get() const
	{
		return m_parent.asRef();
	}

	TParent* operator->() const
	{
		return &m_parent;
	}

private:
	TParent& m_parent;

	// To keep parent alive when the parent is not owning this object
	std::optional<std::shared_ptr<TParent>> m_parent_ref;
};
}
