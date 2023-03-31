
#include "routines/add_source.hpp"

template <>
void add_source(ScalarField<TMP::MemSpaceHost> &p, const float_type src, size_t ix, size_t iz,
                TMP::ExecutionSpaceSerial)
{
    assert(ix < p.get_nx());
    assert(iz < p.get_nz());
    p.get_ptr()[iz * p.get_nx() + ix] = src;
}