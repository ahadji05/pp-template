cmake_minimum_required(VERSION 3.10)
project(Test_TMP)

set(CMAKE_CXX_STANDARD 11)

include(../cmake/select_backend.cmake)
include(../cmake/select_build_mode.cmake)

include_directories(
    "../include/"
    "./"
)

add_executable(main_exe main.cpp)
add_subdirectory(../source build)

target_link_libraries(main_exe
    host_source_code
)