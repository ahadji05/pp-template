
#include "ppt/routines/add_source.hpp"

template <>
void add_source(ScalarField<ppt::MemSpaceHost> &p, const float_type src, size_t ix, size_t iz,
                ppt::ExecutionSpaceSerial)
{
    assert(ix < p.get_nx());
    assert(iz < p.get_nz());
    p.get_ptr()[iz * p.get_nx() + ix] = src;
}

#if defined(PPT_ENABLE_OPENMP_BACKEND)
template <>
void add_source(ScalarField<ppt::MemSpaceHost> &p, const float_type src, size_t ix, size_t iz,
                ppt::ExecutionSpaceOpenMP)
{
    // there is no parallelism to exploit; just call the Serial implementation
    add_source(p, src, ix, iz, ppt::ExecutionSpaceSerial());
}
#endif // PPT_ENABLE_OPENMP_BACKEND
