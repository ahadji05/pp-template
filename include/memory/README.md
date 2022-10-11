## The memory-spaces provide 5 basic routines for data/memory management operations assuming a host-device system is targeted.

```cpp
// 1. Allocate memory.
template <typename value_t, typename length_t>
return_t allocate(value_t **ptr, length_t n_elems);
 
// 2. Free memory.
template <typename value_t> 
return_t release(value_t *ptr);

// 3. Copy data within this memory-space.
template <typename value_t, typename length_t>
return_t copy(value_t *to, value_t *from, length_t n_elems);

// 4. Copy data from host memory-space to this memory-space.
template <typename value_t, typename length_t>
return_t copyFromHost(value_t *to, value_t *from, length_t n_elems);

// 5. Copy data from this memory-space to host memory-space.
template <typename value_t, typename length_t>
return_t copyToHost(value_t *to, value_t *from, length_t n_elems);
```

```cpp
/**
 * The template types associated with the memory operations are:
 * value_t:  value-type of the data.
 * lenght_t: length-type or index-type (it must be a std::integral type).
 * 
 * 
 * The routines always return a message of type return_t:
 * no_error:          returned by all routines when the memory-related operation was successfull.
 * allocation_failed: returned by the allocate routine when the allocation has failed.
 * release_failed:    returned by the release routine when the release has failed.
 * copying_failed:    returned by the copy((From/To)Host) routines when the copying has failed.
 */
```

Example 1: CPU-GPU implementation with compile-time switching to either CUDA or HIP API.
```cpp
/*filename example1.cpp*/

#include "MemorySpacesInc.hpp"

#if defined (TMP_ENABLE_CUDA_BACKEND)
using DeviceSpace = MemSpaceCuda;
#elif defined(TMP_ENABLE_HIP_BACKEND)
using DeviceSpace = MemSpaceHip;
#endif

MemSpaceHost::message msg;
DeviceSpace::message  msgd;

int main()
{
    float *A, *d_A;
    size_t N(100);
    
    msg = MemSpaceHost::allocate(&A, N);
    msgd = DeviceSpace::allocate(&d_A, N);

    // Perform data initialization, pre-processing, etc., on A

    msgd = DeviceSpace::copyFromHost(d_A, A, N);

    // Perform further operations on d_A

    msgd = DeviceSpace::copyToHost(A, d_A, N);

    msgd = DeviceSpace::release(d_A);
    msg = MemSpaceHost::release(A);

    return 0;
}
```