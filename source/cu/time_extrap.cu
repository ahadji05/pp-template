
#include "cuda_config.hpp"
#include "routines/time_extrap.hpp"

__global__ void fd_time_extrap_kernel(float_type *pnew_data, float_type *p_data, float_type *pold_data,
                                      float_type *pxx_data, float_type *pzz_data, float_type *velmodel_data,
                                      float_type dt, float_type dh, size_t nz, size_t nx)
{
    size_t ix = blockDim.x * blockIdx.x + threadIdx.x;
    size_t iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix > nx)
        return;
    if (iz > nz)
        return;

    size_t i = iz * nx + ix;
    pnew_data[i] = (2 * p_data[i] - pold_data[i]) +
                   ((dt * dt) / (dh * dh)) * (velmodel_data[i] * velmodel_data[i]) * (pxx_data[i] + pzz_data[i]);
}

template <>
#if defined(TMP_ENABLE_CUDA_BACKEND)
void fd_time_extrap(ScalarField<TMP::MemSpaceCuda> &pnew, const ScalarField<TMP::MemSpaceCuda> &p,
                    const ScalarField<TMP::MemSpaceCuda> &pold, const ScalarField<TMP::MemSpaceCuda> &pxx,
                    const ScalarField<TMP::MemSpaceCuda> &pzz, const ScalarField<TMP::MemSpaceCuda> &velmodel,
                    float_type dt, float_type dh, TMP::ExecutionSpaceCuda)
#elif defined(TMP_ENABLE_HIP_BACKEND)
void fd_time_extrap(ScalarField<TMP::MemSpaceHip> &pnew, const ScalarField<TMP::MemSpaceHip> &p,
                    const ScalarField<TMP::MemSpaceHip> &pold, const ScalarField<TMP::MemSpaceHip> &pxx,
                    const ScalarField<TMP::MemSpaceHip> &pzz, const ScalarField<TMP::MemSpaceHip> &velmodel,
                    float_type dt, float_type dh, TMP::ExecutionSpaceHip)
#endif
{
    size_t nz = pxx.get_nz();
    size_t nx = pxx.get_nx();

    float_type *pnew_data = pnew.get_ptr();
    float_type *p_data = p.get_ptr();
    float_type *pold_data = pold.get_ptr();
    float_type *pxx_data = pxx.get_ptr();
    float_type *pzz_data = pzz.get_ptr();
    float_type *velmodel_data = velmodel.get_ptr();

    dim3 nThreads(BLOCKDIM_X, BLOCKDIM_Z, 1);
    size_t nBlock_x = nx % BLOCKDIM_X == 0 ? (nx / BLOCKDIM_X) : (1 + nx / BLOCKDIM_X);
    size_t nBlock_z = nz % BLOCKDIM_Z == 0 ? (nz / BLOCKDIM_Z) : (1 + nz / BLOCKDIM_Z);
    dim3 nBlocks(nBlock_x, nBlock_z, 1);

#if defined(TMP_ENABLE_CUDA_BACKEND)
    fd_time_extrap_kernel<<<nBlocks, nThreads>>>(pnew_data, p_data, pold_data, pxx_data, pzz_data, velmodel_data, dt, dh,
                                                nz, nx);
#elif defined(TMP_ENABLE_HIP_BACKEND)
    static_assert(false, "NOT IMPLEMENTED YET");
#endif
}