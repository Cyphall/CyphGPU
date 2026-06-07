#pragma once

#include <memory>

namespace cgpu
{
class Context;
class ContextSession;
class Surface;
class Device;
class DeviceSession;
class Queue;
class VertexInputState;

using ContextPtr = std::shared_ptr<Context>;
using ContextSessionPtr = std::shared_ptr<ContextSession>;
using SurfacePtr = std::shared_ptr<Surface>;
using DevicePtr = std::shared_ptr<Device>;
using DeviceSessionPtr = std::shared_ptr<DeviceSession>;
using QueuePtr = std::shared_ptr<Queue>;
using VertexInputStatePtr = std::shared_ptr<VertexInputState>;
}
