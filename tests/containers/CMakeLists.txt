
FILE(GLOB_RECURSE container_source *.cpp)

add_library(testcontainers
    OBJECT
    ${container_source}
)