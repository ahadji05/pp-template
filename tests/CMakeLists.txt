find_package(GTest REQUIRED)
# TODO: ignore find_gtest.cmake for now, do -DGTest_ROOT=... when configuring
# include(${PPT_TOP_SOURCE_DIR}/cmake/find_gtest.cmake)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

set(PPT_TEST_SOURCES)
append_glob(PPT_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/*.cpp)
append_glob(PPT_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/containers/*.cpp)
append_glob(PPT_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/routines/*.cpp)

add_executable(ppt_test gtest_main.cpp ${PPT_TEST_SOURCES})

target_compile_options(
  ppt_test PRIVATE $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:
                   -Wall -Wextra -Wpedantic>)
target_compile_features(ppt_test PUBLIC cxx_std_11)

target_link_libraries(ppt_test GTest::gtest_main PPT::ppt)
