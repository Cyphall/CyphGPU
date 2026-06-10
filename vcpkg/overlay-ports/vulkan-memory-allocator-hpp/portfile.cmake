set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO YaaZ/VulkanMemoryAllocator-Hpp
    REF "v${VERSION}+3"
    SHA512 14d853962f7410a6495c9a6c0cd7ce8b977e7eca5f4e243c9e442461d25d3e7d7671097e3962a7560af07a9618069eb01072e69a52220e61c21dc0deb52b9b75
    HEAD_REF master
)

file(COPY "${CMAKE_CURRENT_LIST_DIR}/config.cmake.in" DESTINATION "${SOURCE_PATH}")
file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${SOURCE_PATH}")

vcpkg_cmake_configure(
	SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(PACKAGE_NAME "VulkanMemoryAllocatorHpp" CONFIG_PATH "lib/cmake/VulkanMemoryAllocatorHpp")

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
