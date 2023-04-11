
#include "cuda_config.hpp"
#include "routines/stencil.hpp"

__global__ void fd_pxx_kernel(float_type *pxx_data, float_type *p_data, size_t nz, size_t nx)
{
    size_t ix = blockDim.x * blockIdx.x + threadIdx.x;
    size_t iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix < 2 || ix >= nx - 2)
        return;

    pxx_data[iz * nx + ix] = -(1 / 12) * p_data[iz * nx + ix - 2] + (4 / 3) * p_data[iz * nx + ix - 1] -
                             (5 / 2) * p_data[iz * nx + ix] + (4 / 3) * p_data[iz * nx + ix + 1] -
                             (1 / 12) * p_data[iz * nx + ix + 1];
}

__global__ void fd_pzz_kernel(float_type *pzz_data, float_type *p_data, size_t nz, size_t nx)
{
    size_t ix = blockDim.x * blockIdx.x + threadIdx.x;
    size_t iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (iz < 2 || iz >= nz - 2)
        return;

    pzz_data[iz * nx + ix] = -(1 / 12) * p_data[(iz - 2) * nx + ix] + (4 / 3) * p_data[(iz - 1) * nx + ix] -
                             (5 / 2) * p_data[iz * nx + ix] + (4 / 3) * p_data[(iz + 1) * nx + ix] -
                             (1 / 12) * p_data[(iz + 2) * nx + ix];
}

template <>
#if defined(TMP_ENABLE_CUDA_BACKEND)
void fd_pxx(ScalarField<TMP::MemSpaceCuda> &pxx, const ScalarField<TMP::MemSpaceCuda> &p, TMP::ExecutionSpaceCuda)
#elif defined(TMP_ENABLE_HIP_BACKEND)
void fd_pxx(ScalarField<TMP::MemSpaceHip> &pxx, const ScalarField<TMP::MemSpaceHip> &p, TMP::ExecutionSpaceHip)
#endif
{
    assert(pxx.get_nx() == p.get_nx());
    assert(pxx.get_nz() == p.get_nz());

    size_t nz = pxx.get_nz();
    size_t nx = pxx.get_nx();

    float_type *pxx_data = pxx.get_ptr();
    float_type *p_data = p.get_ptr();

    // NOTE FOR POTENTIAL BUG:
    // IF the range of the loop becomes negative,
    // for size_t it wrap-arounds to 18446744073709551615!!!
    assert(nx > 1);

    dim3 nThreads(BLOCKDIM_X, BLOCKDIM_Z, 1);
    size_t nBlock_x = nx % BLOCKDIM_X == 0 ? (nx / BLOCKDIM_X) : (1 + nx / BLOCKDIM_X);
    size_t nBlock_z = nz % BLOCKDIM_Z == 0 ? (nz / BLOCKDIM_Z) : (1 + nz / BLOCKDIM_Z);
    dim3 nBlocks(nBlock_x, nBlock_z, 1);

#if defined(TMP_ENABLE_CUDA_BACKEND)
    fd_pxx_kernel<<<nBlocks, nThreads>>>(pxx_data, p_data, nz, nx);
#elif defined(TMP_ENABLE_HIP_BACKEND)
    static_assert(false, "NOT IMPLEMENTED YET");
#endif
}

template <>
#if defined(TMP_ENABLE_CUDA_BACKEND)
void fd_pzz(ScalarField<TMP::MemSpaceCuda> &pzz, const ScalarField<TMP::MemSpaceCuda> &p, TMP::ExecutionSpaceCuda)
#elif defined(TMP_ENABLE_HIP_BACKEND)
void fd_pzz(ScalarField<TMP::MemSpaceHip> &pzz, const ScalarField<TMP::MemSpaceHip> &p, TMP::ExecutionSpaceHip)
#endif
{
    assert(pzz.get_nx() == p.get_nx());
    assert(pzz.get_nz() == p.get_nz());

    size_t nz = pzz.get_nz();
    size_t nx = pzz.get_nx();

    float_type *pzz_data = pzz.get_ptr();
    float_type *p_data = p.get_ptr();

    // NOTE FOR POTENTIAL BUG:
    // IF the range of the loop becomes negative,
    // for size_t it wrap-arounds to 18446744073709551615!!!
    assert(nz > 1);

    dim3 nThreads(BLOCKDIM_X, BLOCKDIM_Z, 1);
    size_t nBlock_x = nx % BLOCKDIM_X == 0 ? (nx / BLOCKDIM_X) : (1 + nx / BLOCKDIM_X);
    size_t nBlock_z = nz % BLOCKDIM_Z == 0 ? (nz / BLOCKDIM_Z) : (1 + nz / BLOCKDIM_Z);
    dim3 nBlocks(nBlock_x, nBlock_z, 1);

#if defined(TMP_ENABLE_CUDA_BACKEND)
    fd_pzz_kernel<<<nBlocks, nThreads>>>(pzz_data, p_data, nz, nx);
#elif defined(TMP_ENABLE_HIP_BACKEND)
    static_assert(false, "NOT IMPLEMENTED YET");
#endif
}