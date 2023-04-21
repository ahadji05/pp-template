# GLOB_RECURSE mode will traverse all the subdirectories of this
# directory and match the files with extenstion ".cpp"
# The mathced files will be be listed, and parsed to the variable "cpp_source",
# which will be compiled later on and produce the object file "host_source_code",
# for linking with the applications that are build on top of!

# ALL host routines should be stored in a file with extension ".cpp"
FILE(GLOB_RECURSE cpp_source *.cpp)
add_library(host_source_code
    OBJECT
    ${cpp_source}
)

# ALL host routines should be stored in a file with extension ".cpp"
FILE(GLOB_RECURSE cu_source *.cu)
add_library(device_source_code
    OBJECT
    ${cu_source}
)