set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
	OUT_SOURCE_PATH SOURCE_PATH
	REPO GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
	REF "v${VERSION}"
	SHA512 ad6232fa175aea9cfd16f240eed541eb1fb86349aa98c94ae874de85dd53ae78b1db9b293556d8f178c0648e2287f63109caede8595a1ba9ff244c9ddb0b9c48
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
