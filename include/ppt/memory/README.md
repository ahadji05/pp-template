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

Example 1: CPU-GPU implementation with compile-time switching to either CUDA or HIP memory-management calls.
```cpp
#include "MemorySpacesInc.hpp"

using HostSpace = ppt::MemSpaceHost; // alias the host memory-space

#if defined(PPT_ENABLE_CUDA_BACKEND)
using DeviceSpace = ppt::MemSpaceCuda; // alias the device memory-space
#elif defined(PPT_ENABLE_HIP_BACKEND)
using DeviceSpace = ppt::MemSpaceHip; // alias the device memory-space
#endif

HostSpace::message msg;
DeviceSpace::message msgd;

int main()
{
    float *A, *d_A;
    size_t N(100);

    msg = HostSpace::allocate(&A, N);
    msgd = DeviceSpace::allocate(&d_A, N);

    // Perform data initialization, pre-processing, etc., on A

    msgd = DeviceSpace::copyFromHost(d_A, A, N);

    // Perform further operations on d_A

    msgd = DeviceSpace::copyToHost(A, d_A, N);

    msgd = DeviceSpace::release(d_A);
    msg = HostSpace::release(A);

    return 0;
}
```

Example 2: Develop data-structures with generic memory-space, e.g. a custom vector implementation (customVector) that is constructable from HOST std::vector, and safely copy-able.
```cpp
#include "MemorySpacesInc.hpp"
#include <vector>
#include <cassert>

// Generic with respect to memory-space, and value-type.
template<class MemSpace, typename T>
class customVector {
    typename MemSpace::return_t msg;
    
  public:
    customVector();
    customVector(const customVector& v);
    customVector &operator=(const customVector& v);
    customVector(std::vector<T> &v);
    ~customVector();
    
    size_t getN() const { return _N; }
    T * getP() const { return _p; }
    
  private:
    T *_p;
    size_t _N;
};

// Default constructor
template<class MemSpace, typename T>
customVector<MemSpace, T>::customVector()
{
    _N = 0;
    _p = nullptr;
}

// Copy-constructor
template<class MemSpace, typename T>
customVector<MemSpace, T>::customVector(const customVector &v)
{
    _N = v._N;    
    assert( MemSpace::allocate(&_p, _N) == MemSpace::message::no_error );
    assert( MemSpace::copy(_p, v._p, _N) == MemSpace::message::no_error );
}

// Copy-assign operator
template<class MemSpace, typename T>
customVector<MemSpace, T> & customVector<MemSpace, T>::operator=(const customVector &v)
{
    if(this != &v)
    {
        if(_N != v._N)
        {
            _N = v._N;
            assert( MemSpace::release(_p) == MemSpace::message::no_error );
            assert( MemSpace::allocate(&_p, _N) == MemSpace::message::no_error );
        }
        assert( MemSpace::copy(_p, v._p, _N) == MemSpace::message::no_error );
    }
    return *this;
}

// Construct from existing (HOST) std::vector
template<class MemSpace, typename T>
customVector<MemSpace, T>::customVector(std::vector<T> &v)
{
    _N = v.size();    
    assert( MemSpace::allocate(&_p, _N) == MemSpace::message::no_error );
    assert( MemSpace::copyFromHost(_p, v.data(), _N) == MemSpace::message::no_error );
}

// Destructor
template<class MemSpace, typename T>
customVector<MemSpace, T>::~customVector()
{
    assert( MemSpace::release(_p) == MemSpace::message::no_error );
}
```

