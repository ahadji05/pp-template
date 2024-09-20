
#include "ppt/routines/utils.hpp"

template <>
void scale(int nData, float_type *data, float_type scaling_value, 
     [[maybe_unused]] ppt::StreamHost::type stream, ppt::ExecutionSpaceSerial)
{
    for ( int i=0; i < nData; ++i )
        data[i] *= scaling_value;
}

#if defined(PPT_ENABLE_OPENMP_BACKEND)
template <>
void scale(int nData, float_type *data, float_type scaling_value, 
     [[maybe_unused]] ppt::StreamHost::type stream, ppt::ExecutionSpaceOpenMP)
{
    #pragma omp parallel for
    for ( int i=0; i < nData; ++i )
        data[i] *= scaling_value;
}
#endif // PPT_ENABLE_OPENMP_BACKEND
