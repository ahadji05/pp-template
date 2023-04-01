// GENERAL INCLUDES
#include "assert.h"
#include "execution/ExecutionSpacesInc.hpp"

#define _USE_MATH_DEFINES
#include "math.h"

// SELECT FLOATING-POINT PRECISION (DATA-TYPE)
using float_type = float;

// SELECT EXECUTION-SPACE BASED ON COMPILE-TIME CHOICE
#if defined(TMP_ENABLE_CUDA_BACKEND)
using exec_space = TMP::ExecutionSpaceCuda;
#elif defined(TMP_ENABLE_HIP_BACKEND)
using exec_space = TMP::ExecutionSpaceHip;
#elif defined(TMP_ENABLE_OPENMP_BACKEND)
using exec_space = TMP::ExecutionSpaceOpenMP;
#else
using exec_space = TMP::ExecutionSpaceSerial;
#endif

// DEFINE memo_space AS THE accessible_space OF THE SELECTED EXECUTION-SPACE
using memo_space = typename exec_space::accessible_space;