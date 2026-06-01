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
}
