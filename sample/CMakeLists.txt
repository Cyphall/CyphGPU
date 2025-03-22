# ---- Dependency: glfw3 ----

find_package(glfw3 CONFIG REQUIRED)

# ---- Library ----

add_executable(CyphGPU_sample)

target_sources(CyphGPU_sample PRIVATE
	"src/cpp/CyphGPU_sample/Main.cpp"
)

target_sources(CyphGPU_sample PUBLIC FILE_SET HEADERS BASE_DIRS "src/cpp" FILES
)

set_target_properties(CyphGPU_sample PROPERTIES CXX_STANDARD 20)
set_target_properties(CyphGPU_sample PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(CyphGPU_sample PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(CyphGPU_sample PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
set_target_properties(CyphGPU_sample PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
set_target_properties(CyphGPU_sample PROPERTIES COMPILE_WARNING_AS_ERROR $<CONFIG:Release>)

# Enable all warnings
if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
	target_compile_options(CyphGPU_sample PRIVATE /W4)
elseif (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
	target_compile_options(CyphGPU_sample PRIVATE -Wall -Wextra -Wpedantic)
else ()
	message(FATAL_ERROR "Unknown compiler frontend variant: ${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}")
endif ()

# ---- Dependency: CyphGPU ----

target_link_libraries(CyphGPU_sample PRIVATE CyphGPU)

# ---- Dependency: glfw3 ----

target_link_libraries(CyphGPU_sample PRIVATE glfw)
