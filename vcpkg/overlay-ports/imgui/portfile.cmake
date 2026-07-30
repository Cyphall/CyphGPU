vcpkg_from_github(
	OUT_SOURCE_PATH SOURCE_PATH
	REPO ocornut/imgui
	REF "v${VERSION}-docking"
	SHA512 1a5ede24f8358c93a6a012a8961776eccca32dfefe0ea3b88cbbb562a76e6ef6418ab353ed0ec7d59069e5e51918cf2b38b041243cb6edaee0ffd040ccee21c6
	HEAD_REF docking
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