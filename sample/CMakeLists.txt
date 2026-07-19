# ---- Dependency: boostorg/scope ----

find_package(boost_scope CONFIG REQUIRED)

# ---- Dependency: glfw3 ----

find_package(glfw3 CONFIG REQUIRED)

# ---- Dependency: nothings/stb ----

find_package(Stb REQUIRED)

# ---- Library ----

add_executable(CyphGPU_sample)

target_sources(CyphGPU_sample PRIVATE
	"src/cpp/CyphGPU_sample/Main.cpp"
)

target_sources(CyphGPU_sample PUBLIC FILE_SET HEADERS BASE_DIRS "src/cpp" FILES
	"src/cpp/CyphGPU_sample/texture.h"
	"src/cpp/CyphGPU_sample/mesh.h"
)

include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/utils/shader.cmake")
cgpu_target_shader_pipelines(CyphGPU_sample shaders "src/slang"
	"src/slang/shader.slang"
)

set_target_properties(CyphGPU_sample PROPERTIES CXX_STANDARD 23)
set_target_properties(CyphGPU_sample PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(CyphGPU_sample PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(CyphGPU_sample PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
set_target_properties(CyphGPU_sample PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
set_target_properties(CyphGPU_sample PROPERTIES CXX_SCAN_FOR_MODULES OFF)

# Enable all warnings + warning-as-error in Release
if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
	target_compile_options(CyphGPU_sample PRIVATE /W4)
	target_compile_options(CyphGPU_sample PRIVATE $<$<CONFIG:Release>:/WX>)
elseif (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
	target_compile_options(CyphGPU_sample PRIVATE -Wall -Wextra -Wpedantic)
	target_compile_options(CyphGPU_sample PRIVATE $<$<CONFIG:Release>:-Werror>)
else ()
	message(FATAL_ERROR "Unknown compiler frontend variant: ${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}")
endif ()

install(
	TARGETS CyphGPU_sample
)

file(GLOB COPYRIGHT_FILES "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/*/copyright")
foreach (COPYRIGHT_FILE IN LISTS COPYRIGHT_FILES)
	cmake_path(GET COPYRIGHT_FILE PARENT_PATH COPYRIGHT_PORT_DIR)
	cmake_path(GET COPYRIGHT_PORT_DIR FILENAME COPYRIGHT_PORT)
	if (NOT COPYRIGHT_PORT MATCHES "^vcpkg-")
		install(
			FILES "${COPYRIGHT_FILE}"
			DESTINATION "share/CyphGPU_sample/third-party-licenses"
			RENAME "${COPYRIGHT_PORT}.txt"
		)
	endif ()
endforeach ()

install(
	FILES "${CMAKE_CURRENT_SOURCE_DIR}/vkcube-license.txt"
	DESTINATION "share/CyphGPU_sample/third-party-licenses"
	RENAME "vulkan-tools.txt"
)

# ---- Dependency: CyphGPU ----

target_link_libraries(CyphGPU_sample PRIVATE CyphGPU)

# ---- Dependency: boostorg/scope ----

target_link_libraries(CyphGPU_sample PRIVATE Boost::scope)

# ---- Dependency: glfw3 ----

target_link_libraries(CyphGPU_sample PRIVATE glfw)

# ---- Dependency: nothings/stb ----

target_include_directories(CyphGPU_sample PRIVATE ${Stb_INCLUDE_DIR})
