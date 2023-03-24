// GENERAL INCLUDES
#include "assert.h"

// SELECT FLOATING-POINT PRECISION (DATA-TYPE)
#include "half.hpp"
using float_type = half_float::half;

// SELECT MEMORY-SPACE BASED ON COMPILE-TIME CHOICE
#include "memory/MemorySpacesInc.hpp"
#if defined(TMP_ENABLE_CUDA_BACKEND)
using memo_space = TMP::MemSpaceCuda;
#elif defined(TMP_ENABLE_HIP_BACKEND)
using memo_space = TMP::MemSpaceHip;
#else
using memo_space = TMP::MemSpaceHost;
#endif

// DEFINE VECTOR-TYPE vector_type based on: memo_space and float_type
#include "containers/Vector.hpp"
using vector_type = TMP::Vector<float_type, memo_space>;