
#include "ppt/routines/stencil.hpp"

template <>
void fd_pxx(ScalarField<ppt::MemSpaceHost> &pxx, const ScalarField<ppt::MemSpaceHost> &p, ppt::ExecutionSpaceSerial)
{
    assert(pxx.get_nx() == p.get_nx());
    assert(pxx.get_nz() == p.get_nz());

    size_t nz = pxx.get_nz();
    size_t nx = pxx.get_nx();

    float_type *pxx_data = pxx.get_ptr();
    float_type *p_data   = p.get_ptr();

    // NOTE FOR POTENTIAL BUG:
    // IF the range of the loop becomes negative,
    // for size_t it wrap-arounds to 18446744073709551615!!!
    assert(nx > 1);

    float_type c0 = -5.0 / 2.0;
    float_type c1 = 4.0 / 3.0;
    float_type c2 = -1.0 / 12.0;

    for (size_t iz(0); iz < nz; ++iz)
        for (size_t ix(2); ix < nx - 2; ++ix)
        {
            size_t i = iz * nx + ix;
            pxx_data[i] =
                c2 * p_data[i - 2] + c1 * p_data[i - 1] + c0 * p_data[i] + c1 * p_data[i + 1] + c2 * p_data[i + 2];
        }
}

template <>
void fd_pzz(ScalarField<ppt::MemSpaceHost> &pzz, const ScalarField<ppt::MemSpaceHost> &p, ppt::ExecutionSpaceSerial)
{
    assert(pzz.get_nx() == p.get_nx());
    assert(pzz.get_nz() == p.get_nz());

    size_t nz = pzz.get_nz();
    size_t nx = pzz.get_nx();

    float_type *pzz_data = pzz.get_ptr();
    float_type *p_data   = p.get_ptr();

    // NOTE FOR POTENTIAL BUG:
    // IF the range of the loop becomes negative,
    // for size_t it wrap-arounds to 18446744073709551615!!!
    assert(nz > 1);

    float_type c0 = -5.0 / 2.0;
    float_type c1 = 4.0 / 3.0;
    float_type c2 = -1.0 / 12.0;

    for (size_t iz(2); iz < nz - 2; ++iz)
        for (size_t ix(0); ix < nx; ++ix)
            pzz_data[iz * nx + ix] = c2 * p_data[(iz - 2) * nx + ix] + c1 * p_data[(iz - 1) * nx + ix] +
                                     c0 * p_data[iz * nx + ix] + c1 * p_data[(iz + 1) * nx + ix] +
                                     c2 * p_data[(iz + 2) * nx + ix];
}

#if defined(PPT_ENABLE_OPENMP_BACKEND)
template <>
void fd_pxx(ScalarField<ppt::MemSpaceHost> &pxx, const ScalarField<ppt::MemSpaceHost> &p, ppt::ExecutionSpaceOpenMP)
{
    assert(pxx.get_nx() == p.get_nx());
    assert(pxx.get_nz() == p.get_nz());

    size_t nz = pxx.get_nz();
    size_t nx = pxx.get_nx();

    float_type *pxx_data = pxx.get_ptr();
    float_type *p_data   = p.get_ptr();

    // NOTE FOR POTENTIAL BUG:
    // IF the range of the loop becomes negative,
    // for size_t it wrap-arounds to 18446744073709551615!!!
    assert(nx > 1);

    float_type c0 = -5.0 / 2.0;
    float_type c1 = 4.0 / 3.0;
    float_type c2 = -1.0 / 12.0;

#pragma omp parallel for
    for (size_t iz = 0; iz < nz; ++iz)
        for (size_t ix = 2; ix < nx - 2; ++ix)
        {
            size_t i = iz * nx + ix;
            pxx_data[i] =
                c2 * p_data[i - 2] + c1 * p_data[i - 1] + c0 * p_data[i] + c1 * p_data[i + 1] + c2 * p_data[i + 2];
        }
}

template <>
void fd_pzz(ScalarField<ppt::MemSpaceHost> &pzz, const ScalarField<ppt::MemSpaceHost> &p, ppt::ExecutionSpaceOpenMP)
{
    assert(pzz.get_nx() == p.get_nx());
    assert(pzz.get_nz() == p.get_nz());

    size_t nz = pzz.get_nz();
    size_t nx = pzz.get_nx();

    float_type *pzz_data = pzz.get_ptr();
    float_type *p_data   = p.get_ptr();

    // NOTE FOR POTENTIAL BUG:
    // IF the range of the loop becomes negative,
    // for size_t it wrap-arounds to 18446744073709551615!!!
    assert(nz > 1);

    float_type c0 = -5.0 / 2.0;
    float_type c1 = 4.0 / 3.0;
    float_type c2 = -1.0 / 12.0;

#pragma omp parallel for
    for (size_t iz = 2; iz < nz - 2; ++iz)
        for (size_t ix = 0; ix < nx; ++ix)
            pzz_data[iz * nx + ix] = c2 * p_data[(iz - 2) * nx + ix] + c1 * p_data[(iz - 1) * nx + ix] +
                                     c0 * p_data[iz * nx + ix] + c1 * p_data[(iz + 1) * nx + ix] +
                                     c2 * p_data[(iz + 2) * nx + ix];
}
#endif // PPT_ENABLE_OPENMP_BACKEND
