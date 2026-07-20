#pragma once

#include <boost/container/static_vector.hpp>
#include <functional>
#include <vector>

namespace cgpu
{
template<class T, class THash = std::hash<T>>
void hashCombine(size_t& seed, const T& value)
{
	// https://www.boost.org/doc/libs/1_91_0/libs/container_hash/doc/html/hash.html#notes_hash_combine

	seed += 0x9e3779b9 + THash{}(value);

	seed ^= seed >> 32;
	seed *= 0xe9846af9b1a615d;
	seed ^= seed >> 32;
	seed *= 0xe9846af9b1a615d;
	seed ^= seed >> 28;
}

template<class T>
struct ContainerHash
{
	std::size_t operator()(const T& key) const noexcept
	{
		size_t seed = 0;
		for (const auto& element : key)
		{
			hashCombine(seed, element);
		}
		return seed;
	}
};
}

template<class T, class TAllocator>
struct std::hash<std::vector<T, TAllocator>> : cgpu::ContainerHash<std::vector<T, TAllocator>>
{};

template<class T, size_t Capacity, class TOptions>
struct std::hash<boost::container::static_vector<T, Capacity, TOptions>> : cgpu::ContainerHash<boost::container::static_vector<T, Capacity, TOptions>>
{};
