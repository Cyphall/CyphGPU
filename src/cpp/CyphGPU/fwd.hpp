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
class Buffer;
class Image;
class Sampler;
class Swapchain;
class VertexInputState;
class PreRasterizationShaderState;
class FragmentShaderState;
class FragmentOutputState;

using ContextPtr = std::shared_ptr<Context>;
using ContextSessionPtr = std::shared_ptr<ContextSession>;
using SurfacePtr = std::shared_ptr<Surface>;
using DevicePtr = std::shared_ptr<Device>;
using DeviceSessionPtr = std::shared_ptr<DeviceSession>;
using QueuePtr = std::shared_ptr<Queue>;
using BufferPtr = std::shared_ptr<Buffer>;
using ImagePtr = std::shared_ptr<Image>;
using SamplerPtr = std::shared_ptr<Sampler>;
using SwapchainPtr = std::shared_ptr<Swapchain>;
using VertexInputStatePtr = std::shared_ptr<VertexInputState>;
using PreRasterizationShaderStatePtr = std::shared_ptr<PreRasterizationShaderState>;
using FragmentShaderStatePtr = std::shared_ptr<FragmentShaderState>;
using FragmentOutputStatePtr = std::shared_ptr<FragmentOutputState>;
}
