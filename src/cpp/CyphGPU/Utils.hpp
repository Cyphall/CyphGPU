#pragma once

#include <string_view>

namespace cgpu
{
struct StringHash : std::hash<std::string_view>
{
	using is_transparent = void;
};

struct StringLess : std::less<std::string_view>
{
	using is_transparent = void;
};

struct StringEqualTo : std::equal_to<std::string_view>
{
	using is_transparent = void;
};

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
}
