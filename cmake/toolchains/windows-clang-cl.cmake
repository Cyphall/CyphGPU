set(CMAKE_C_COMPILER clang-cl.exe)
set(CMAKE_CXX_COMPILER clang-cl.exe)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_C_FLAGS_INIT "/Zc:__cplusplus /utf-8 /arch:AVX2")
set(CMAKE_CXX_FLAGS_INIT "/Zc:__cplusplus /utf-8 /arch:AVX2")
