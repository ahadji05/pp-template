cmake_minimum_required(VERSION 3.5)
project(ppt LANGUAGES CXX)

set(PPT_TOP_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(PPT_TOP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(PPT_VERSION_MAJOR 00)
set(PPT_VERSION_MINOR 01)
set(PPT_VERSION_PATCH 00)
set(PPT_VERSION
    "${PPT_VERSION_MAJOR}.${PPT_VERSION_MINOR}.${PPT_VERSION_PATCH}")
math(
  EXPR
  ppt_VERSION
  "${PPT_VERSION_MAJOR} * 10000 + ${PPT_VERSION_MINOR} * 100 + ${PPT_VERSION_PATCH}"
)
message(STATUS "Configuring PPT v${PPT_VERSION}")

message(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
cmake_policy(SET CMP0074 NEW)

include(cmake/cmake_utils.cmake)

set(PPT_ENABLE_TESTS
    OFF
    CACHE BOOL "Whether to build tests. Default: OFF")
set(PPT_ENABLE_DOCS
    OFF
    CACHE BOOL "Whether to build docs. Default: OFF")

add_subdirectory(src)

if(PPT_ENABLE_TESTS)
  message(STATUS "Configuring unit tests")
  add_subdirectory(tests)
  message(STATUS "Configuring unit tests - done")
endif()

if(PPT_ENABLE_DOCS)
  message(STATUS "Configuring docs")
  include(cmake/cmake_doxygen.cmake)
  configure_docs()
  message(STATUS "Configuring docs - done")
endif()

ppt_postprocess()
