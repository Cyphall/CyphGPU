vcpkg_from_github(
	OUT_SOURCE_PATH SOURCE_PATH
	REPO ocornut/imgui
	REF "v${VERSION}"
	SHA512 1a8fc7e4d7fe8926289ed9598f39dd5b601baffa3b2a7a0889ed0f9a8f252c85710f4ba65b2a6801bb5b46a17d1fd30b5542e11f67b8989c6640b498ef68bb2d
	HEAD_REF master
)

file(COPY "${CMAKE_CURRENT_LIST_DIR}/config.cmake.in" DESTINATION "${SOURCE_PATH}")
file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${SOURCE_PATH}")

vcpkg_cmake_configure(
	SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(CONFIG_PATH "lib/cmake/imgui")

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")