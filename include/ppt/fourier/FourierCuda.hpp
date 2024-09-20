/**
 * @file
 *
 * @author  Andreas Hadjigeorgiou, The Cyprus Institute,
 *          Personal-site: https://ahadji05.github.io,
 *          E-mail: a.hadjigeorgiou@cyi.ac.cy
 *
 * @copyright 2022 CaSToRC (The Cyprus Institute), Delphi Consortium (TU Delft)
 *
 * @version 1.0
 *
 * @section LICENCE
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#ifndef PPT_FOURIER_CUDA
#define PPT_FOURIER_CUDA

#include "ppt/fourier/FourierBase.hpp"
#include "ppt/stream/StreamInc.hpp"
#include "ppt/definitions.hpp"
#include "cuda_runtime.h"
#include "cufft.h"
#include "ppt/routines/utils.hpp"

namespace ppt 
{

class FourierCuda : public FourierBase {

  public:
    using type = cufftHandle;

    // alloc memory and create plan for batched 1d ffts
    static int fftplan1d_float( cufftHandle **handle, int signalSize, int nBatch ) {
      *handle = new cufftHandle();
      cufftPlan1d( *handle, signalSize, CUFFT_C2C, nBatch );
      return 0;
    }

    // destroy plan and free memory
    static void fftplan_destroy( cufftHandle *handle ){
      cufftDestroy(*handle);
      delete handle;
      handle = nullptr;
    }

    // attach fft plan to a stream
    static int fftplan_attach_stream( cufftHandle *handle, ppt::StreamCuda::type *stream ) {
      cufftSetStream( *handle, *stream );
      return 0;
    }

    // execute plan forward
    static int fftplan_exec_forward( cufftHandle *handle, float *data_in, float *data_out ) {
      cufftResult_t r = cufftExecC2C( *handle, (cufftComplex*)data_in, (cufftComplex*)data_out, CUFFT_FORWARD );
      if ( r != cudaSuccess ) return 1;
      return 0;
    }

    // execute plan backward
    static int fftplan_exec_backward( cufftHandle *handle, float *data_in, float *data_out ) {
      cufftExecC2C( *handle, (cufftComplex*)data_in, (cufftComplex*)data_out, CUFFT_INVERSE );
      return 0;
    }

    // fft scale
    static int fft_scale( int64_t N, float *data, float scaling_value, ppt::StreamCuda::type stream ) {
      scale( N, data, scaling_value, stream, ppt::ExecutionSpaceCuda());
      return 0;
    }
};

} // namespace ppt

#endif