#pragma once

#include <CyphGPU/detail/BumpMemoryResource.hpp>

#include <cstddef>
#include <flat_map>
#include <flat_set>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cgpu::detail
{
template<class T = std::byte>
class BumpAllocator
{
public:
	using value_type = T;

	BumpAllocator(BumpMemoryResource& resource):
		m_resource{&resource}
	{}

	BumpAllocator(const BumpAllocator&) = default;

	template<class TOther>
	BumpAllocator(const BumpAllocator<TOther>& other):
		m_resource{other.m_resource}
	{}

	BumpAllocator& operator=(const BumpAllocator&) = delete;

	T* allocate(size_t count)
	{
		return static_cast<T*>(m_resource->allocate(count * sizeof(T), alignof(T)));
	}

	void deallocate(T* ptr, size_t count)
	{
		m_resource->deallocate(ptr, count * sizeof(T), alignof(T));
	}

	bool operator==(const BumpAllocator& other) const = default;

private:
	template<class>
	friend class BumpAllocator;

	BumpMemoryResource* m_resource;
};

template<class T>
using BumpVector = std::vector<T, BumpAllocator<T>>;

template<class TKey, class TValue, class TCompare = std::less<TKey>>
using BumpMap = std::map<TKey, TValue, TCompare, BumpAllocator<std::pair<const TKey, TValue>>>;

template<class TKey, class TValue, class THash = std::hash<TKey>, class TKeyEqual = std::equal_to<TKey>>
using BumpUnorderedMap = std::unordered_map<TKey, TValue, THash, TKeyEqual, BumpAllocator<std::pair<const TKey, TValue>>>;

template<class TKey, class TValue, class TCompare = std::less<TKey>>
using BumpFlatMap = std::flat_map<TKey, TValue, TCompare, BumpVector<TKey>, BumpVector<TValue>>;

template<class TKey, class TCompare = std::less<TKey>>
using BumpSet = std::set<TKey, TCompare, BumpAllocator<TKey>>;

template<class TKey, class THash = std::hash<TKey>, class TKeyEqual = std::equal_to<TKey>>
using BumpUnorderedSet = std::unordered_set<TKey, THash, TKeyEqual, BumpAllocator<TKey>>;

template<class TKey, class TCompare = std::less<TKey>>
using BumpFlatSet = std::flat_set<TKey, TCompare, BumpVector<TKey>>;

template<class T>
using BumpList = std::list<T, BumpAllocator<T>>;
}
