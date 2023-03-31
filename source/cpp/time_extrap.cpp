
#include "routines/time_extrap.hpp"

template <>
void fd_time_extrap(ScalarField<TMP::MemSpaceHost> &pnew, const ScalarField<TMP::MemSpaceHost> &p,
                    const ScalarField<TMP::MemSpaceHost> &pold, const ScalarField<TMP::MemSpaceHost> &pxx,
                    const ScalarField<TMP::MemSpaceHost> &pzz, const ScalarField<TMP::MemSpaceHost> &velmodel,
                    float_type dt, float_type dh, TMP::ExecutionSpaceSerial)
{
    size_t nz = pxx.get_nz();
    size_t nx = pxx.get_nx();

    float_type *pnew_data = pnew.get_ptr();
    float_type *p_data = p.get_ptr();
    float_type *pold_data = pold.get_ptr();
    float_type *pxx_data = pxx.get_ptr();
    float_type *pzz_data = pzz.get_ptr();
    float_type *velmodel_data = velmodel.get_ptr();

    for (size_t iz(0); iz < nz; ++iz)
        for (size_t ix(0); ix < nx; ++ix)
        {
            size_t i = iz * nx + ix;
            pnew_data[i] = (2 * p_data[i] - pold_data[i]) + ((dt * dt) / (dh * dh)) *
                                                                (velmodel_data[i] * velmodel_data[i]) *
                                                                (pxx_data[i] + pzz_data[i]);
        }
}