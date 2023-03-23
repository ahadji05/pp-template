cmake_minimum_required(VERSION 3.10)
project(Test_TMP)

set(CMAKE_CXX_STANDARD 11)

include(../cmake/select_backend.cmake)
include(../cmake/select_build_mode.cmake)
include(../cmake/find_gtest.cmake)

include_directories(
    "../include/"
    "./"
)

add_subdirectory(algorithms)
add_subdirectory(containers)
add_subdirectory(routines)
add_subdirectory(../source build)

add_executable(gtest_all gtest_main.cpp)

target_link_libraries(gtest_all
    GTest::gtest_main
    testcontainers
    testalgorithms
    testroutines
    host_source_code
)