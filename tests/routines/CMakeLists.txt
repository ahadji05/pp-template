
FILE(GLOB_RECURSE routines_source *.cpp)

add_library(testroutines
    OBJECT
    ${routines_source}
)