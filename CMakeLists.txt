cmake_minimum_required(VERSION 3.28)

# ---- Project ----

project(
	CyphGPU
	VERSION 1.0.0
	LANGUAGES CXX
)

# ---- Include guards ----

if (PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
	message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif ()

# ---- Options ----

option(CYPHGPU_INSTALL "Install CyphGPU." OFF)
option(CYPHGPU_ENABLE_SAMPLE "Enable CyphGPU sample." OFF)
option(CYPHGPU_ENABLE_TRACY "Enable tracy." OFF)

string(COMPARE EQUAL "${CMAKE_SYSTEM_NAME}" "Windows" CYPHGPU_IS_WINDOWS)
string(COMPARE EQUAL "${CMAKE_SYSTEM_NAME}" "Darwin" CYPHGPU_IS_MACOS)
string(COMPARE EQUAL "${CMAKE_SYSTEM_NAME}" "Linux" CYPHGPU_IS_LINUX)

option(CYPHGPU_ENABLE_PLATFORM_WIN32 "Enable Win32 platform capabilities." ${CYPHGPU_IS_WINDOWS})
option(CYPHGPU_ENABLE_PLATFORM_METAL "Enable Metal platform capabilities." ${CYPHGPU_IS_MACOS})
option(CYPHGPU_ENABLE_PLATFORM_XCB "Enable X11 Xcb platform capabilities." ${CYPHGPU_IS_LINUX})
option(CYPHGPU_ENABLE_PLATFORM_XLIB "Enable X11 Xlib platform capabilities." ${CYPHGPU_IS_LINUX})
option(CYPHGPU_ENABLE_PLATFORM_WAYLAND "Enable Wayland platform capabilities." ${CYPHGPU_IS_LINUX})

# ---- Dependency: glm ----

find_package(glm CONFIG REQUIRED)

# ---- Dependency: Vulkan-Headers ----

find_package(VulkanHeaders CONFIG REQUIRED)

# ---- Dependency: GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator ----

find_package(VulkanMemoryAllocator CONFIG REQUIRED)

# ---- Dependency: Neargye/magic_enum ----

find_package(magic_enum CONFIG REQUIRED)

# ---- Dependency: boostorg/optional ----

find_package(boost_optional CONFIG REQUIRED)

# ---- Dependency: gabime/spdlog ----

find_package(spdlog CONFIG REQUIRED)

# ---- Dependency: wolfpld/tracy ----

find_package(Tracy CONFIG REQUIRED)

# ---- Dependency: vector-of-bool/cmrc ----

find_package(CMakeRC CONFIG REQUIRED)

# ---- Dependency: half ----

find_path(HALF_INCLUDE_DIRS "half.hpp" REQUIRED)

# ---- Dependency: martinus/unordered_dense ----

find_package(unordered_dense CONFIG REQUIRED)

# ---- Library ----

add_library(CyphGPU)

target_sources(CyphGPU PRIVATE
	"src/cpp/CyphGPU/Utils.cpp"
	"src/cpp/CyphGPU/Context.cpp"
	"src/cpp/CyphGPU/ContextSession.cpp"
	"src/cpp/CyphGPU/Surface.cpp"
	"src/cpp/CyphGPU/Device.cpp"
	"src/cpp/CyphGPU/DeviceSession.cpp"
	"src/cpp/CyphGPU/Queue.cpp"
	"src/cpp/CyphGPU/VertexInputState.cpp"
	"src/cpp/CyphGPU/PreRasterizationShaderState.cpp"
	"src/cpp/CyphGPU/FragmentShaderState.cpp"
	"src/cpp/CyphGPU/FragmentOutputState.cpp"
	"src/cpp/CyphGPU/detail/VMA.cpp"
	"src/cpp/CyphGPU/Buffer.cpp"
	"src/cpp/CyphGPU/Image.cpp"
	"src/cpp/CyphGPU/Sampler.cpp"
	"src/cpp/CyphGPU/Swapchain.cpp"
	"src/cpp/CyphGPU/CommandContext.cpp"
	"src/cpp/CyphGPU/CommandContextSlot.cpp"
	"src/cpp/CyphGPU/CommandRecorder.cpp"
	"src/cpp/CyphGPU/ComputeShaderState.cpp"
	"src/cpp/CyphGPU/ComputePassContext.cpp"
	"src/cpp/CyphGPU/Resource.cpp"
	"src/cpp/CyphGPU/GraphicsPassContext.cpp"
	"src/cpp/CyphGPU/detail/ShaderChainBuilder.cpp"
	"src/cpp/CyphGPU/ShaderBundle.cpp"
	"src/cpp/CyphGPU/PassContext.cpp"
)

target_sources(CyphGPU PUBLIC FILE_SET HEADERS BASE_DIRS "src/cpp" FILES
	"src/cpp/CyphGPU/Utils.hpp"
	"src/cpp/CyphGPU/fwd.hpp"
	"src/cpp/CyphGPU/Context.hpp"
	"src/cpp/CyphGPU/ContextSession.hpp"
	"src/cpp/CyphGPU/Surface.hpp"
	"src/cpp/CyphGPU/Device.hpp"
	"src/cpp/CyphGPU/detail/DynamicFeatureChain.hpp"
	"src/cpp/CyphGPU/DeviceSession.hpp"
	"src/cpp/CyphGPU/Queue.hpp"
	"src/cpp/CyphGPU/VertexInputState.hpp"
	"src/cpp/CyphGPU/PreRasterizationShaderState.hpp"
	"src/cpp/CyphGPU/FragmentShaderState.hpp"
	"src/cpp/CyphGPU/FragmentOutputState.hpp"
	"src/cpp/CyphGPU/MemoryType.hpp"
	"src/cpp/CyphGPU/Buffer.hpp"
	"src/cpp/CyphGPU/Image.hpp"
	"src/cpp/CyphGPU/Sampler.hpp"
	"src/cpp/CyphGPU/Swapchain.hpp"
	"src/cpp/CyphGPU/HashExt.hpp"
	"src/cpp/CyphGPU/CommandContext.hpp"
	"src/cpp/CyphGPU/CommandContextSlot.hpp"
	"src/cpp/CyphGPU/CommandRecorder.hpp"
	"src/cpp/CyphGPU/ComputeShaderState.hpp"
	"src/cpp/CyphGPU/ComputePassContext.hpp"
	"src/cpp/CyphGPU/Resource.hpp"
	"src/cpp/CyphGPU/GraphicsPassContext.hpp"
	"src/cpp/CyphGPU/detail/ShaderChainBuilder.hpp"
	"src/cpp/CyphGPU/ShaderBundle.hpp"
	"src/cpp/CyphGPU/ShaderTypes.hpp"
	"src/cpp/CyphGPU/PassContext.hpp"
	"src/cpp/CyphGPU/detail/BumpAllocator.hpp"
	"src/cpp/CyphGPU/detail/BumpMemoryResource.hpp"
)

target_precompile_headers(CyphGPU PRIVATE <vulkan/vulkan.hpp>)

set_target_properties(CyphGPU PROPERTIES CXX_STANDARD 23)
set_target_properties(CyphGPU PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(CyphGPU PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(CyphGPU PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
set_target_properties(CyphGPU PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
set_target_properties(CyphGPU PROPERTIES CXX_SCAN_FOR_MODULES OFF)

# Enable all warnings + warning-as-error in Release
if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
	target_compile_options(CyphGPU PRIVATE /W4)
	target_compile_options(CyphGPU PRIVATE $<$<CONFIG:Release>:/WX>)
elseif (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
	target_compile_options(CyphGPU PRIVATE -Wall -Wextra -Wpedantic)
	target_compile_options(CyphGPU PRIVATE $<$<CONFIG:Release>:-Werror>)
else ()
	message(FATAL_ERROR "Unknown compiler frontend variant: ${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}")
endif ()

# Vulkan-Hpp and VMA-Hpp heavily depend on aliasing
if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
	target_compile_options(CyphGPU PUBLIC -fno-strict-aliasing)
endif ()

# ---- Install ----

if (CYPHGPU_INSTALL)
	install(
		TARGETS CyphGPU
		EXPORT CyphGPUTargets
		FILE_SET HEADERS
	)

	install(
		EXPORT CyphGPUTargets
		DESTINATION "lib/cmake/CyphGPU"
	)

	include(CMakePackageConfigHelpers)
	configure_package_config_file(
		"${CMAKE_CURRENT_SOURCE_DIR}/cmake/install/Config.cmake.in"
		"${CMAKE_CURRENT_BINARY_DIR}/CyphGPUConfig.cmake"
		INSTALL_DESTINATION "lib/cmake/CyphGPU"
	)
	write_basic_package_version_file(
		"${CMAKE_CURRENT_BINARY_DIR}/CyphGPUConfigVersion.cmake"
		COMPATIBILITY SameMinorVersion
	)

	install(
		FILES
			"${CMAKE_CURRENT_BINARY_DIR}/CyphGPUConfig.cmake"
			"${CMAKE_CURRENT_BINARY_DIR}/CyphGPUConfigVersion.cmake"
		DESTINATION "lib/cmake/CyphGPU"
	)

	install(
		FILES
			"${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/shader.cmake"
		DESTINATION "lib/cmake/CyphGPU"
	)
endif ()

# ---- Dependency: glm ----

target_link_libraries(CyphGPU PUBLIC glm::glm)

target_compile_definitions(CyphGPU PUBLIC GLM_ENABLE_EXPERIMENTAL)
target_compile_definitions(CyphGPU PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)

# ---- Dependency: Vulkan-Headers ----

target_link_libraries(CyphGPU PUBLIC Vulkan::Headers)

target_compile_definitions(CyphGPU PUBLIC VK_NO_PROTOTYPES)
target_compile_definitions(CyphGPU PUBLIC VULKAN_HPP_NO_DEFAULT_DISPATCHER)
target_compile_definitions(CyphGPU PUBLIC VULKAN_HPP_NO_SETTERS)
target_compile_definitions(CyphGPU PUBLIC VULKAN_HPP_NO_STRUCT_CONSTRUCTORS)
target_compile_definitions(CyphGPU PUBLIC NOMINMAX)

target_compile_definitions(CyphGPU PUBLIC $<$<BOOL:${CYPHGPU_ENABLE_PLATFORM_WIN32}>:VK_USE_PLATFORM_WIN32_KHR>)
target_compile_definitions(CyphGPU PUBLIC $<$<BOOL:${CYPHGPU_ENABLE_PLATFORM_METAL}>:VK_USE_PLATFORM_METAL_EXT>)
target_compile_definitions(CyphGPU PUBLIC $<$<BOOL:${CYPHGPU_ENABLE_PLATFORM_XCB}>:VK_USE_PLATFORM_XCB_KHR>)
target_compile_definitions(CyphGPU PUBLIC $<$<BOOL:${CYPHGPU_ENABLE_PLATFORM_XLIB}>:VK_USE_PLATFORM_XLIB_KHR>)
target_compile_definitions(CyphGPU PUBLIC $<$<BOOL:${CYPHGPU_ENABLE_PLATFORM_WAYLAND}>:VK_USE_PLATFORM_WAYLAND_KHR>)

# ---- Dependency: GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator ----

target_link_libraries(CyphGPU PUBLIC VulkanMemoryAllocator)

target_compile_definitions(CyphGPU PUBLIC VMA_STATIC_VULKAN_FUNCTIONS=0)
target_compile_definitions(CyphGPU PUBLIC VMA_DYNAMIC_VULKAN_FUNCTIONS=1)

# ---- Dependency: Neargye/magic_enum ----

target_link_libraries(CyphGPU PUBLIC magic_enum::magic_enum)

# ---- Dependency: boostorg/optional ----

target_link_libraries(CyphGPU PUBLIC Boost::optional)

# ---- Dependency: gabime/spdlog ----

target_link_libraries(CyphGPU PUBLIC spdlog::spdlog)

# ---- Dependency: wolfpld/tracy ----

if (CYPHGPU_ENABLE_TRACY)
	target_link_libraries(CyphGPU PUBLIC Tracy::TracyClient)
else ()
	target_include_directories(CyphGPU PUBLIC $<TARGET_PROPERTY:Tracy::TracyClient,INTERFACE_INCLUDE_DIRECTORIES>)
endif ()

# ---- Dependency: vector-of-bool/cmrc ----

target_link_libraries(CyphGPU PUBLIC $<BUILD_INTERFACE:cmrc::base>)

# ---- Dependency: half ----

target_include_directories(CyphGPU PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIRS}>)

# ---- Dependency: martinus/unordered_dense ----

target_link_libraries(CyphGPU PUBLIC unordered_dense::unordered_dense)

# ---- Sample ----

if (CYPHGPU_ENABLE_SAMPLE)
	add_subdirectory("sample")
endif ()
