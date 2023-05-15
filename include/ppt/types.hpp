#ifndef PPT_TYPES_HPP
#define PPT_TYPES_HPP

// GENERAL INCLUDES
#include "assert.h"
#include "ppt/execution/ExecutionSpacesInc.hpp"

#define _USE_MATH_DEFINES
#include "math.h"

// SELECT FLOATING-POINT PRECISION (DATA-TYPE)
using float_type = double;

// SELECT EXECUTION-SPACE BASED ON COMPILE-TIME CHOICE
#if defined(PPT_ENABLE_CUDA_BACKEND)
using exec_space = ppt::ExecutionSpaceCuda;
#elif defined(PPT_ENABLE_HIP_BACKEND)
using exec_space = ppt::ExecutionSpaceHip;
#elif defined(PPT_ENABLE_OPENMP_BACKEND)
using exec_space = ppt::ExecutionSpaceOpenMP;
#else
using exec_space = ppt::ExecutionSpaceSerial;
#endif

// DEFINE memo_space AS THE accessible_space OF THE SELECTED EXECUTION-SPACE
using memo_space = typename exec_space::accessible_space;

#endif // PPT_TYPES_HPP
