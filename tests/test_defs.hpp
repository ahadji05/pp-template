#ifndef TMP_TEST_DEFS_HPP
#define TMP_TEST_DEFS_HPP

#include "gtest/gtest.h"
const float test_margin = 0.00001;

#include "memory/MemorySpacesInc.hpp"

#if defined(TMP_ENABLE_CUDA_BACKEND)
using memo_space = TMP::MemSpaceCuda;
#elif defined(TMP_ENABLE_HIP_BACKEND)
using memo_space = TMP::MemSpaceHip;
#else
using memo_space = TMP::MemSpaceHost;
#endif

#endif