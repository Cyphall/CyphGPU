vcpkg_from_github(
	OUT_SOURCE_PATH SOURCE_PATH
	REPO wolfpld/tracy
	REF "v${VERSION}"
	SHA512 d6d07db668e62e2c4fb476b549243c240434613554e99bd68b6446b56b92e6cec606246186a02964ed9a223b5ccdac55546185510cd9f6fda05d458314fb05c1
	HEAD_REF master
)

vcpkg_cmake_configure(
	SOURCE_PATH "${SOURCE_PATH}"
	OPTIONS
	-DTRACY_ENABLE=ON
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(PACKAGE_NAME "Tracy" CONFIG_PATH "lib/cmake/Tracy")

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
