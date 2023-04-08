cmake_minimum_required(VERSION 3.10)
project(WAVE_SIMULATON)

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

if(TMP_ENABLE_CUDA_BACKEND)
target_link_libraries(main_exe
    device_source_code
)
endif()