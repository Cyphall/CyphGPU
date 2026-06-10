set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
	OUT_SOURCE_PATH SOURCE_PATH
	REPO GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
	REF "v${VERSION}"
	SHA512 deb5902ef8db0e329fbd5f3f4385eb0e26bdd9f14f3a2334823fb3fe18f36bc5d235d620d6e5f6fe3551ec3ea7038638899db8778c09f6d5c278f5ff95c3344b
	HEAD_REF master
)

file(COPY "${CMAKE_CURRENT_LIST_DIR}/config.cmake.in" DESTINATION "${SOURCE_PATH}")
file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${SOURCE_PATH}")

vcpkg_cmake_configure(
	SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(PACKAGE_NAME "VulkanMemoryAllocator" CONFIG_PATH "lib/cmake/VulkanMemoryAllocator")

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")
