
#include "cuda_config.hpp"
#include "ppt/routines/add_source.hpp"

__global__ void add_source_kernel(float_type *p, float_type src, size_t ix, size_t nx, size_t iz)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > 0) return;

    p[iz * nx + ix] = src;
}

#if defined(PPT_ENABLE_CUDA_BACKEND)
template <>
void add_source(ScalarField<ppt::MemSpaceCuda> &p, const float_type src, size_t ix, size_t iz, ppt::ExecutionSpaceCuda)
#elif defined(PPT_ENABLE_HIP_BACKEND)
template <>
void add_source(ScalarField<ppt::MemSpaceHip> &p, const float_type src, size_t ix, size_t iz, ppt::ExecutionSpaceHip)
#endif
{
    assert(ix < p.get_nx());
    assert(iz < p.get_nz());

    size_t nx = p.get_nx();
#if defined(PPT_ENABLE_CUDA_BACKEND)
    add_source_kernel<<<1, 1>>>(p.get_ptr(), src, ix, nx, iz);
    cudaDeviceSynchronize();
#elif defined(PPT_ENABLE_HIP_BACKEND)
    hipLaunchKernelGGL(add_source_kernel, 1, 1, 0, NULL, p.get_ptr(), src, ix, nx, iz);
    hipDeviceSynchronize();
#endif
}
