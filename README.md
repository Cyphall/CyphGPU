[![Build](https://github.com/Cyphall/CyphGPU/actions/workflows/build.yaml/badge.svg)](https://github.com/Cyphall/CyphGPU/actions/workflows/build.yaml)

# CyphGPU

A rendering backend over modern Vulkan for my other projects.

> [!IMPORTANT]  
> Not production-ready and probably never will.

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
cmake --build --preset [PRESET_NAME] --target CyphGPU

# To install (optional)
cmake --build --preset [PRESET_NAME] --target install
```
