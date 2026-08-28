#pragma once

#include <CyphGPU/detail/BumpMemoryResource.hpp>

#include <ankerl/unordered_dense.h>
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

	explicit BumpAllocator(BumpMemoryResource& resource):
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

	void deallocate(T*, size_t)
	{
	}

	bool operator==(const BumpAllocator& other) const = default;

private:
	template<class>
	friend class BumpAllocator;

	BumpMemoryResource* m_resource;
};

template<class T>
struct BumpDeleter
{
	BumpDeleter() = default;

	template<class TOther>
	requires(std::is_convertible_v<TOther*, T*>)
	BumpDeleter(const BumpDeleter<TOther>&)
	{}

	void operator()(T* ptr) const
	{
		std::destroy_at(ptr);
	}
};

template<
	class T>
using BumpVector = std::vector<T, BumpAllocator<T>>;

template<
	class TKey,
	class TValue,
	class TCompare = std::less<TKey>>
using BumpMap = std::map<TKey, TValue, TCompare, BumpAllocator<std::pair<const TKey, TValue>>>;

template<
	class TKey,
	class TValue,
	class THash = ankerl::unordered_dense::hash<TKey>,
	class TKeyEqual = std::equal_to<TKey>>
using BumpUnorderedMap = std::unordered_map<TKey, TValue, THash, TKeyEqual, BumpAllocator<std::pair<const TKey, TValue>>>;

template<
	class TKey,
	class TValue,
	class TCompare = std::less<TKey>>
using BumpFlatMap = std::flat_map<TKey, TValue, TCompare, BumpVector<TKey>, BumpVector<TValue>>;

template<
	class TKey,
	class TCompare = std::less<TKey>>
using BumpSet = std::set<TKey, TCompare, BumpAllocator<TKey>>;

template<
	class TKey,
	class THash = ankerl::unordered_dense::hash<TKey>,
	class TKeyEqual = std::equal_to<TKey>>
using BumpUnorderedSet = std::unordered_set<TKey, THash, TKeyEqual, BumpAllocator<TKey>>;

template<
	class TKey,
	class TCompare = std::less<TKey>>
using BumpFlatSet = std::flat_set<TKey, TCompare, BumpVector<TKey>>;

template<
	class T>
using BumpList = std::list<T, BumpAllocator<T>>;

template<
	class TKey,
	class TValue,
	class THash = ankerl::unordered_dense::hash<TKey>,
	class TKeyEqual = std::equal_to<TKey>,
	class TBucket = ankerl::unordered_dense::bucket_type::standard,
	class TBucketContainer = ankerl::unordered_dense::detail::default_container_t>
using BumpDenseUnorderedMap = ankerl::unordered_dense::map<TKey, TValue, THash, TKeyEqual, BumpAllocator<std::pair<TKey, TValue>>, TBucket, TBucketContainer>;

template<
	class TKey,
	class TValue,
	class THash = ankerl::unordered_dense::hash<TKey>,
	class TKeyEqual = std::equal_to<TKey>>
using BumpSegmentedUnorderedMap = ankerl::unordered_dense::segmented_map<TKey, TValue, THash, TKeyEqual, BumpAllocator<std::pair<TKey, TValue>>>;

template<
	class TKey,
	class THash = ankerl::unordered_dense::hash<TKey>,
	class TKeyEqual = std::equal_to<TKey>,
	class TBucket = ankerl::unordered_dense::bucket_type::standard,
	class TBucketContainer = ankerl::unordered_dense::detail::default_container_t>
using BumpDenseUnorderedSet = ankerl::unordered_dense::set<TKey, THash, TKeyEqual, BumpAllocator<TKey>, TBucket, TBucketContainer>;

template<
	class TKey,
	class THash = ankerl::unordered_dense::hash<TKey>,
	class TKeyEqual = std::equal_to<TKey>>
using BumpSegmentedUnorderedSet = ankerl::unordered_dense::segmented_set<TKey, THash, TKeyEqual, BumpAllocator<TKey>>;

template<
	class T,
	std::size_t MaxSegmentSizeBytes = 4096>
using BumpSegmentedVector = ankerl::unordered_dense::segmented_vector<T, BumpAllocator<T>, MaxSegmentSizeBytes>;

template<
	class T>
using BumpUniquePtr = std::unique_ptr<T, BumpDeleter<T>>;

template<class T, class... TArgs>
constexpr BumpUniquePtr<T> makeBumpUnique(detail::BumpMemoryResource& bump_memory, TArgs&&... args)
{
	void* memory = bump_memory.allocate(sizeof(T), alignof(T));
	return BumpUniquePtr<T>{std::construct_at(static_cast<T*>(memory), std::forward<TArgs>(args)...)};
}
}
