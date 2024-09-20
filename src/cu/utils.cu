
#include "cuda_config.hpp"
#include "ppt/routines/utils.hpp"

__global__ void scale_kernel( float_type *vec_data, int nx, float_type scaling_value ) {
    size_t ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix > nx) return;
    vec_data[ix] *= scaling_value;
}

#if defined(PPT_ENABLE_CUDA_BACKEND)
template <>
void scale(int nData, float_type *data, float_type scaling_value, 
    ppt::StreamCuda::type stream, ppt::ExecutionSpaceCuda)
#elif defined(PPT_ENABLE_HIP_BACKEND)
template <>
void scale(int nData, float_type *data, float_type scaling_value, 
    ppt::StreamHip::type stream, ppt::ExecutionSpaceHip)
#endif
{
    dim3 nThreads(BLOCKDIM_X, 1, 1);
    size_t nBlock_x = nData % BLOCKDIM_X == 0 ? (nData / BLOCKDIM_X) : (1 + nData / BLOCKDIM_X);
    dim3 nBlocks(nBlock_x, 1, 1);

#if defined(PPT_ENABLE_CUDA_BACKEND)
    scale_kernel<<<nBlocks, nThreads, 0, stream>>>(data, nData, scaling_value);
#elif defined(PPT_ENABLE_HIP_BACKEND)
    hipLaunchKernelGGL(scale_kernel, nBlocks, nThreads, 0, stream, data, nData, scaling_value);
#endif
}