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
#ifndef PPT_STREAM_CUDA
#define PPT_STREAM_CUDA

#include "ppt/stream/StreamBase.hpp"
#include "ppt/definitions.hpp"
#include "cuda_runtime.h"

namespace ppt 
{

class StreamCuda : public StreamBase {

  public:
    using type = cudaStream_t;

    // create a new stream
    static void create( cudaStream_t **pStream ) {
        if (*pStream)
            StreamCuda::destroy(*pStream);
        *pStream = new cudaStream_t();
        cudaError_t status = cudaStreamCreate( *pStream );
        if (status != cudaSuccess) throw std::runtime_error("cudaStreamCreate failed!");
    }

    // destroy an existing stream
    static void destroy( cudaStream_t *pStream ){
        cudaError_t status = cudaStreamDestroy(*pStream);
        if (status != cudaSuccess) throw std::runtime_error("cudaStreamDestroy failed!");
        delete pStream;
        pStream = nullptr;
    }

    // synchronize an existing stream
    static void sync( cudaStream_t *pStream ){
        cudaError_t status = cudaStreamSynchronize(*pStream);
        if (status != cudaSuccess) throw std::runtime_error("cudaStreamSynchronize failed!");
    }
};

} // namespace ppt

#endif