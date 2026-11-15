vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY ONLY_DYNAMIC_CRT)
set(VCPKG_BUILD_TYPE release)
set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)
set(VCPKG_POLICY_EMPTY_INCLUDE_FOLDER enabled)

if (VCPKG_TARGET_IS_WINDOWS)
	vcpkg_download_distfile(
		ARCHIVE
		URLS "https://github.com/shader-slang/slang/releases/download/v${VERSION}/slang-${VERSION}-windows-x86_64.zip"
		FILENAME "slang-${VERSION}-windows-x86_64.zip"
		SHA512 cac45e9f659ae31cd9a4dbdd3be5c062bf97c1e354fb99212742c3e77866bc519326e63b392a067d74829d3faaabc28271efe615129e5acf8386edd863ba363e
	)
elseif (VCPKG_TARGET_IS_OSX)
	vcpkg_download_distfile(
		ARCHIVE
		URLS "https://github.com/shader-slang/slang/releases/download/v${VERSION}/slang-${VERSION}-macos-aarch64.zip"
		FILENAME "slang-${VERSION}-macos-aarch64.zip"
		SHA512 fdffcdb5070b676d25c23c118791ede6fa6db046e8eb5a299c3e91a595e3716b75026f21993199120f771a9ac4034c80d8cd6d221b75b1cd5c896d3d4cd874f0
	)
elseif (VCPKG_TARGET_IS_LINUX)
	vcpkg_download_distfile(
		ARCHIVE
		URLS "https://github.com/shader-slang/slang/releases/download/v${VERSION}/slang-${VERSION}-linux-x86_64.zip"
		FILENAME "slang-${VERSION}-linux-x86_64.zip"
		SHA512 6277a36cb7e377767031cfef9239b545adddffb52c33a68aa549e7d62aeacc2d7a07cef98390767ba527afeebf74f15645440189ea31fbf7573a941b9e70e54a
	)
endif ()

vcpkg_extract_source_archive(
	SOURCE_PATH
	ARCHIVE "${ARCHIVE}"
	NO_REMOVE_ONE_LEVEL
)

set(VCPKG_FIXUP_MACHO_RPATH OFF)

file(GLOB SLANGC "${SOURCE_PATH}/*/slangc*")
file(GLOB SLANGD "${SOURCE_PATH}/*/slangd*")
file(COPY ${SLANGC} ${SLANGD} DESTINATION "${CURRENT_PACKAGES_DIR}/tools/${PORT}")

file(GLOB SLANG_COMPILER "${SOURCE_PATH}/*/*slang-compiler*")
file(COPY ${SLANG_COMPILER} DESTINATION "${CURRENT_PACKAGES_DIR}/tools/${PORT}")

file(GLOB SLANG_GLSLANG "${SOURCE_PATH}/*/*slang-glslang*")
file(COPY ${SLANG_GLSLANG} DESTINATION "${CURRENT_PACKAGES_DIR}/tools/${PORT}")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
