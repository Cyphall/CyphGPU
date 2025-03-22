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
		return {m_owningPtr.lock(), static_cast<TDerived*>(this)};
	}

	std::shared_ptr<const TDerived> asRef() const
	{
		return {m_owningPtr.lock(), static_cast<const TDerived*>(this)};
	}

protected:
	template<class, class TParent>
	requires std::derived_from<TParent, DependencyObject<TParent>>
	friend class DependencyObjectChild;

	// Self-owned constructor
	explicit DependencyObject():
		m_owningPtr{m_weakThis}
	{
	}

	// External-owned constructor
	explicit DependencyObject(std::weak_ptr<void>& externalOwningPtr):
		m_owningPtr{externalOwningPtr}
	{
	}

	// Similar to weak_this in std::enable_shared_from_this
	// Will be empty for external-owned objects
	std::weak_ptr<void> m_weakThis;

	// For independent objects, points to our m_weakThis
	// For external-owned objects, points to the m_owningPtr of the owning object
	std::weak_ptr<void>& m_owningPtr;
};

template<class TDerived, class TParent>
requires std::derived_from<TParent, DependencyObject<TParent>>
class DependencyObjectChild : public DependencyObject<TDerived>
{
public:
	std::shared_ptr<TParent> getParent() const
	{
		return m_parent.asRef();
	}

protected:
	explicit DependencyObjectChild(std::shared_ptr<TParent> parent):
		DependencyObject<TDerived>{},
		m_parent{*parent},
		m_parentRef{parent}
	{
	}

	explicit DependencyObjectChild(TParent& parent):
		DependencyObject<TDerived>{parent.m_owningPtr},
		m_parent{parent}
	{
	}

	TParent& m_parent;

private:
	// To keep parent alive when this is self-owned
	std::optional<std::shared_ptr<TParent>> m_parentRef;
};
}
