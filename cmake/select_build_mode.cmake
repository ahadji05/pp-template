message("Setting up build mode...")

if(TMP_DEBUG_MEMORY_MANAGE)
    message("Building tests using debug memory mode...")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTMP_DEBUG_MEMORY_MANAGE")
endif()