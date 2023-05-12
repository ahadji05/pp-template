cmake_minimum_required(VERSION 3.18)
project(
  wave_simulation
  VERSION 0.0.1
  LANGUAGES CXX)

message(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
cmake_policy(SET CMP0074 NEW)

include_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

find_package(ppt REQUIRED)

add_executable(main_exe main.cpp)

target_compile_options(
  main_exe
  PUBLIC $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>: -Wall -Wextra
  -Wpedantic>)
target_compile_features(main_exe PUBLIC cxx_std_11)

target_link_libraries(main_exe PPT::ppt)
