#include "assert.h"
#include "memory/MemorySpacesInc.hpp"

#include "half.hpp"
using float_type = half_float::half;

#if defined(TMP_ENABLE_CUDA_BACKEND)
using memo_space = TMP::MemSpaceCuda;
#elif defined(TMP_ENABLE_HIP_BACKEND)
using memo_space = TMP::MemSpaceHip;
#else
using memo_space = TMP::MemSpaceHost;
#endif