# This will include Compiler options as defined during configuration
configure_file(${PPT_TOP_SOURCE_DIR}/cmake/PPT_config.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/ppt/PPT_config.hpp)

set(PPT_SOURCES)
append_glob(PPT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp)

if(PPT_ENABLE_CUDA)
  append_glob(PPT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cu/*.cu)
endif()

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
  ppt PUBLIC $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>: -Wall
             -Wextra -Wpedantic>)
target_compile_features(ppt PUBLIC cxx_std_11)

if(PPT_ENABLE_OPENMP)
  target_link_libraries(ppt OpenMP::OpenMP_CXX)
endif()

if(PPT_ENABLE_CUDA)
  # We need to explicitly state that we need all CUDA files in the PPT library
  # to be built with -dc as the member functions could be called by other
  # libraries and executables
  set_target_properties(ppt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(ppt PUBLIC CUDA::cudart CUDA::cufft)
endif()

if(PPT_ENABLE_HIP)
  # TODO
endif()
