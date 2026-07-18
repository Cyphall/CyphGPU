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
- VCPKG_ROOT env variable set to a vcpkg install

> [!NOTE]  
> When building the sample app, macOS and Linux may need some additional system dependencies.

### Steps:

```bash
# To build (required)
cmake --preset [PRESET_NAME]
cmake --build --preset [PRESET_NAME] --target CyphGPU

# To install (optional)
cmake --build --preset [PRESET_NAME] --target install
```

## Third party licenses

[KhronosGroup/Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) is licensed under the [MIT License](https://spdx.org/licenses/MIT.html).<br/>
[GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) is licensed under the [MIT License](https://spdx.org/licenses/MIT.html).<br/>
[g-truc/glm](https://github.com/g-truc/glm) is licensed under the [MIT License](https://spdx.org/licenses/MIT.html).<br/>
[Neargye/magic_enum](https://github.com/Neargye/magic_enum) is licensed under the [MIT License](https://spdx.org/licenses/MIT.html).<br/>
[boostorg/boost](https://github.com/boostorg/boost) is licensed under the [Boost Software License 1.0](https://spdx.org/licenses/BSL-1.0.html).<br/>
[gabime/spdlog](https://github.com/gabime/spdlog) is licensed under the [MIT License](https://spdx.org/licenses/MIT.html).<br/>
[glfw/glfw](https://github.com/glfw/glfw) is licensed under the [zlib License](https://spdx.org/licenses/Zlib.html).<br/>
[wolfpld/tracy](https://github.com/wolfpld/tracy) is licensed under the [BSD 3-Clause "New" or "Revised" License](https://spdx.org/licenses/BSD-3-Clause.html).<br/>
[vector-of-bool/cmrc](https://github.com/vector-of-bool/cmrc) is licensed under the [MIT License](https://spdx.org/licenses/MIT.html).<br/>
[half](https://half.sourceforge.net/) is licensed under the [MIT License](https://spdx.org/licenses/MIT.html).<br/>
[shader-slang/slang](https://github.com/shader-slang/slang) is licensed under the [Apache License 2.0](https://spdx.org/licenses/Apache-2.0.html) with [LLVM Exception](https://spdx.org/licenses/LLVM-exception.html).<br/>
