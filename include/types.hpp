// GENERAL INCLUDES
#include "assert.h"

#define _USE_MATH_DEFINES
#include "math.h"

// SELECT FLOATING-POINT PRECISION (DATA-TYPE)
using float_type = float;

// SELECT MEMORY-SPACE BASED ON COMPILE-TIME CHOICE
#include "memory/MemorySpacesInc.hpp"
#if defined(TMP_ENABLE_CUDA_BACKEND)
using memo_space = TMP::MemSpaceCuda;
#elif defined(TMP_ENABLE_HIP_BACKEND)
using memo_space = TMP::MemSpaceHip;
#else
using memo_space = TMP::MemSpaceHost;
#endif