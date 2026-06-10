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

# ---- Dependency: YaaZ/VulkanMemoryAllocator-Hpp ----

find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(unofficial-vulkan-memory-allocator-hpp CONFIG REQUIRED)

# ---- Dependency: Neargye/magic_enum ----

find_package(magic_enum CONFIG REQUIRED)

# ---- Dependency: boostorg/optional ----

find_package(boost_optional CONFIG REQUIRED)

# ---- Dependency: gabime/spdlog ----

find_package(spdlog CONFIG REQUIRED)

# ---- Library ----

add_library(CyphGPU)

target_sources(CyphGPU PRIVATE
	"src/cpp/CyphGPU/Utils.cpp"
	"src/cpp/CyphGPU/Context/Context.cpp"
	"src/cpp/CyphGPU/ContextSession/ContextSession.cpp"
	"src/cpp/CyphGPU/Surface/Surface.cpp"
	"src/cpp/CyphGPU/Device/Device.cpp"
	"src/cpp/CyphGPU/DeviceSession/DeviceSession.cpp"
	"src/cpp/CyphGPU/Queue/Queue.cpp"
	"src/cpp/CyphGPU/Pipeline/VertexInputState.cpp"
	"src/cpp/CyphGPU/VMA.cpp"
)

target_sources(CyphGPU PUBLIC FILE_SET HEADERS BASE_DIRS "src/cpp" FILES
	"src/cpp/CyphGPU/Utils.hpp"
	"src/cpp/CyphGPU/fwd.hpp"
	"src/cpp/CyphGPU/Context/Context.hpp"
	"src/cpp/CyphGPU/ContextSession/ContextSession.hpp"
	"src/cpp/CyphGPU/Surface/Surface.hpp"
	"src/cpp/CyphGPU/Device/Device.hpp"
	"src/cpp/CyphGPU/Device/DynamicFeatureChain.hpp"
	"src/cpp/CyphGPU/DeviceSession/DeviceSession.hpp"
	"src/cpp/CyphGPU/Queue/Queue.hpp"
	"src/cpp/CyphGPU/Pipeline/VertexInputState.hpp"
	"src/cpp/CyphGPU/MemoryType.hpp"
)

target_precompile_headers(CyphGPU PUBLIC <vulkan/vulkan.hpp>)

set_target_properties(CyphGPU PROPERTIES CXX_STANDARD 20)
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

# ---- Dependency: YaaZ/VulkanMemoryAllocator-Hpp ----

target_link_libraries(CyphGPU PUBLIC GPUOpen::VulkanMemoryAllocator)
target_link_libraries(CyphGPU PUBLIC unofficial::VulkanMemoryAllocator-Hpp::VulkanMemoryAllocator-Hpp)

target_compile_definitions(CyphGPU PUBLIC VMA_STATIC_VULKAN_FUNCTIONS=0)
target_compile_definitions(CyphGPU PUBLIC VMA_DYNAMIC_VULKAN_FUNCTIONS=1)

# ---- Dependency: Neargye/magic_enum ----

target_link_libraries(CyphGPU PUBLIC magic_enum::magic_enum)

# ---- Dependency: boostorg/optional ----

target_link_libraries(CyphGPU PUBLIC Boost::optional)

# ---- Dependency: gabime/spdlog ----

target_link_libraries(CyphGPU PUBLIC spdlog::spdlog)

# ---- Sample ----

if (CYPHGPU_ENABLE_SAMPLE)
	add_subdirectory("sample")
endif ()
