cmake_minimum_required(VERSION 3.28)

project(VulkanMemoryAllocatorHpp)

add_library(VulkanMemoryAllocatorHpp INTERFACE)

target_sources(VulkanMemoryAllocatorHpp PUBLIC FILE_SET HEADERS BASE_DIRS "include" FILES
	"include/vk_mem_alloc.hpp"
	"include/vk_mem_alloc_enums.hpp"
	"include/vk_mem_alloc_funcs.hpp"
	"include/vk_mem_alloc_handles.hpp"
	"include/vk_mem_alloc_imported.hpp"
	"include/vk_mem_alloc_raii.hpp"
	"include/vk_mem_alloc_static_assertions.hpp"
	"include/vk_mem_alloc_structs.hpp"
	"include/vk_mem_alloc_to_string.hpp"
)

find_package(VulkanMemoryAllocator CONFIG REQUIRED)
target_link_libraries(VulkanMemoryAllocatorHpp INTERFACE VulkanMemoryAllocator)

install(
	TARGETS VulkanMemoryAllocatorHpp
	EXPORT VulkanMemoryAllocatorHppTargets
	FILE_SET HEADERS
)

install(
	EXPORT VulkanMemoryAllocatorHppTargets
	DESTINATION "lib/cmake/VulkanMemoryAllocatorHpp"
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/config.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/VulkanMemoryAllocatorHppConfig.cmake"
	INSTALL_DESTINATION "lib/cmake/VulkanMemoryAllocatorHpp"
)

install(
	FILES
	"${CMAKE_CURRENT_BINARY_DIR}/VulkanMemoryAllocatorHppConfig.cmake"
	DESTINATION "lib/cmake/VulkanMemoryAllocatorHpp"
)