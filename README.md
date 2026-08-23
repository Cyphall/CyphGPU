[![Build](https://github.com/Cyphall/CyphGPU/actions/workflows/build.yaml/badge.svg)](https://github.com/Cyphall/CyphGPU/actions/workflows/build.yaml)

# CyphGPU

A rendering backend over modern Vulkan for my other projects.

> [!IMPORTANT]  
> Not production-ready and probably never will.

## Features

* Fully bindless (VK_EXT_descriptor_heap, buffer pointers)
* Frame-agnostic (no frames-in-flight concept)
* Async graphics/compute/transfer queues
* Optimal event-based inter-queue automatic sync
* Refcount-based lifetime management
* Separate pipeline stages linked at runtime (VK_EXT_graphics_pipeline_library)
* Ray tracing (VK_KHR_ray_query)
* Built-in compilation and embedding of Slang shaders
* Built-in Tracy CPU & GPU integration
* Built-in ImGui rendering backend

## Build & Install

CyphGPU supports Windows, macOS and Linux.

> [!NOTE]  
> CyphGPU does not bundle a macOS Vulkan driver itself, it needs to be available system-wide like any other Vulkan driver on any other OS.

### Requirements:

- MSVC, Clang or GCC
- CMake 3.28+
- Ninja
- VCPKG_ROOT env variable set to a vcpkg install, or another way of providing dependencies

> [!NOTE]  
> All CMake presets require vcpkg.
>
> When building the sample app, macOS and Linux may need some additional system dependencies.

### Steps:

```bash
# To build (required)
cmake --preset [PRESET_NAME]
cmake --build --preset [PRESET_NAME] --target CyphGPU # or CyphGPU_sample

# To install (optional)
cmake --build --preset [PRESET_NAME] --target install
```
