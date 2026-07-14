vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY ONLY_DYNAMIC_CRT)
set(VCPKG_BUILD_TYPE release)
set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)
set(VCPKG_POLICY_EMPTY_INCLUDE_FOLDER enabled)

if (VCPKG_TARGET_IS_WINDOWS)
	vcpkg_download_distfile(
		ARCHIVE
		URLS "https://github.com/shader-slang/slang/releases/download/v${VERSION}/slang-${VERSION}-windows-x86_64.zip"
		FILENAME "slang-${VERSION}-windows-x86_64.zip"
		SHA512 532d301ee7c900d368e87f04d3d14fa36dd8f84a919b9a94edac12df558940f69b4e6dbe68d9db0a79eff7327e5fd499d96c6abf35fc1c269ce805e6015898f5
	)
elseif (VCPKG_TARGET_IS_OSX)
	vcpkg_download_distfile(
		ARCHIVE
		URLS "https://github.com/shader-slang/slang/releases/download/v${VERSION}/slang-${VERSION}-macos-aarch64.zip"
		FILENAME "slang-${VERSION}-macos-aarch64.zip"
		SHA512 2a0e87d1da2e610212048cd0f6f48ab7392649511688d38f4f6b1d7c8b7b7a4cf89d6244c2adfc3ee40c7ea14d37b149e631f9afadd692559d45c056bb94668e
	)
elseif (VCPKG_TARGET_IS_LINUX)
	vcpkg_download_distfile(
		ARCHIVE
		URLS "https://github.com/shader-slang/slang/releases/download/v${VERSION}/slang-${VERSION}-linux-x86_64.zip"
		FILENAME "slang-${VERSION}-linux-x86_64.zip"
		SHA512 deda14ab92856033f7ffc4303e0250961fe66750ed709a454813d91e7db5c9bb148ad40789e9e7625fb875c840360683eae1f10a859c6be8f41f05b78cd55e85
	)
endif ()

vcpkg_extract_source_archive(
	SOURCE_PATH
	ARCHIVE "${ARCHIVE}"
	NO_REMOVE_ONE_LEVEL
)

set(VCPKG_FIXUP_MACHO_RPATH OFF)

file(GLOB SLANGC "${SOURCE_PATH}/*/slangc*")
file(COPY ${SLANGC} DESTINATION "${CURRENT_PACKAGES_DIR}/tools/${PORT}")

file(GLOB SLANG_COMPILER "${SOURCE_PATH}/*/*slang-compiler*")
file(COPY ${SLANG_COMPILER} DESTINATION "${CURRENT_PACKAGES_DIR}/tools/${PORT}")

file(GLOB SLANG_GLSLANG "${SOURCE_PATH}/*/*slang-glslang*")
file(COPY ${SLANG_GLSLANG} DESTINATION "${CURRENT_PACKAGES_DIR}/tools/${PORT}")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
