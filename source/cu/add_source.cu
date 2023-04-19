
#include "cuda_config.hpp"
#include "routines/add_source.hpp"

__global__ void add_source_kernel(float_type *p, float_type src, size_t ix, size_t nx, size_t iz)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > 0)
        return;
    p[iz * nx + ix] = src;
}

#if defined(TMP_ENABLE_CUDA_BACKEND)
template <>
void add_source(ScalarField<TMP::MemSpaceCuda> &p, const float_type src, size_t ix, size_t iz, TMP::ExecutionSpaceCuda)
#elif defined(TMP_ENABLE_HIP_BACKEND)
template <>
void add_source(ScalarField<TMP::MemSpaceHip> &p, const float_type src, size_t ix, size_t iz, TMP::ExecutionSpaceHip)
#endif
{
    assert(ix < p.get_nx());
    assert(iz < p.get_nz());

    size_t nx = p.get_nx();
#if defined(TMP_ENABLE_CUDA_BACKEND)
    add_source_kernel<<<1, 1>>>(p.get_ptr(), src, ix, nx, iz);
#elif defined(TMP_ENABLE_HIP_BACKEND)
    static_assert(false, "NOT IMPLEMENTED YET");
#endif
}
