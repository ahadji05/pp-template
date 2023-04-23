include(${PPT_TOP_SOURCE_DIR}/cmake/select_backend.cmake)
enable_backends()

# This will include Compiler options as defined during configuration
configure_file(${PPT_TOP_SOURCE_DIR}/cmake/PPT_config.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/ppt/PPT_config.hpp)

set(PPT_SOURCES)
append_glob(PPT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp)

add_library(ppt ${PPT_SOURCES})
# Add alias so that library can be used inside the build tree, e.g. when testing
add_library(PPT::ppt ALIAS ppt)

target_include_directories(
  ppt
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${PPT_TOP_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(
  ppt PRIVATE $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>: -Wall
              -Wextra -Wpedantic>)
target_compile_features(ppt PRIVATE cxx_std_11)

if(PPT_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(ppt OpenMP::OpenMP_CXX)
endif()

if(PPT_ENABLE_CUDA)
  # TODO: link with cuda
endif()

if(PPT_ENABLE_HIP)
  # TODO
endif()
