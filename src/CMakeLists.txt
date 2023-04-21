macro(APPEND_GLOB VAR)
  file(GLOB LOCAL_TMP_VAR ${ARGN})
  list(APPEND ${VAR} ${LOCAL_TMP_VAR})
endmacro()

# This will include Compiler options as defined during configuration
configure_file(${PPT_TOP_SOURCE_DIR}/cmake/PPT_config.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/ppt/PPT_config.hpp)

set(PPT_SOURCES)
append_glob(PPT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp)

add_library(PPT ${PPT_SOURCES})
# Add alias so that library can be used inside the build tree, e.g. when testing
add_library(PPT::PPT ALIAS PPT)

target_include_directories(
  PPT
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${PPT_TOP_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(
  PPT PRIVATE $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>: -Wall
              -Wextra -Wpedantic>)
target_compile_features(PPT PRIVATE cxx_std_11)
