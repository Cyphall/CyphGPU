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

using ContextRef = std::shared_ptr<Context>;
using ContextSessionRef = std::shared_ptr<ContextSession>;
using SurfaceRef = std::shared_ptr<Surface>;
using DeviceRef = std::shared_ptr<Device>;
using DeviceSessionRef = std::shared_ptr<DeviceSession>;
using QueueRef = std::shared_ptr<Queue>;
}
