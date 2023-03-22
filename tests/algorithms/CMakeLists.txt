
FILE(GLOB_RECURSE algorithms_source *.cpp)

add_library(testalgorithms
    OBJECT
    ${algorithms_source}
)