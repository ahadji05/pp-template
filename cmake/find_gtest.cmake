message("Setting up Google-test library...")

find_package(GTest CONFIG)

if(NOT GTest_FOUND)
    message(STATUS "find_package could not find GTest - Downloading GTest")
    include(FetchContent)
    FetchContent_Declare(
        googletest

        # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
    )
endif()