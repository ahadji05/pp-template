macro(ENABLE_BACKENDS)
  set(PPT_ENABLE_OPENMP
    OFF
    CACHE BOOL "Whether to enable OpenMP backend. Default: OFF")
  set(PPT_ENABLE_CUDA
    OFF
    CACHE BOOL "Whether to enable Cuda backend. Default: OFF")
  set(PPT_ENABLE_HIP
    OFF
    CACHE BOOL "Whether to enable HIP backend. Default: OFF")

  if(PPT_ENABLE_CUDA)
    set(PPT_ENABLE_CUDA_BACKEND ON)
    message(STATUS "Configuring with ExecutionSpaceCUDA.")
  elseif(PPT_ENABLE_HIP)
    set(PPT_ENABLE_HIP_BACKEND ON)
    message(STATUS "Configuring with ExecutionSpaceHIP.")
  elseif(PPT_ENABLE_OPENMP)
    set(PPT_ENABLE_OPENMP_BACKEND ON)
    message(STATUS "Configuring with ExecutionSpaceOpenMP.")
  else()
    message(STATUS "Configuring with ExecutionSpaceSerial.")
  endif()
endmacro()
