#pragma once

#include <memory>

namespace cgpu
{
class Context;
class ContextSession;

using ContextRef = std::shared_ptr<Context>;
using ContextSessionRef = std::shared_ptr<ContextSession>;
}