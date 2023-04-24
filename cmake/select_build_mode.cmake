message("Setting up build mode...")

if(PPT_DEBUG_MEMORY_MANAGE)
    message("Building tests using debug memory mode...")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPPT_DEBUG_MEMORY_MANAGE")
endif()