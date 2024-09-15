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
#ifndef PPT_STREAM_HIP
#define PPT_STREAM_HIP

#include "ppt/stream/StreamBase.hpp"
#include "ppt/definitions.hpp"
#include "hip/hip_runtime.h"

namespace ppt 
{

class StreamHip : public StreamBase {

  public:
    using type = hipStream_t;

    // create a new stream
    static void create( hipStream_t **pStream ) {
        if (*pStream)
            StreamHip::destroy(*pStream);
        *pStream = new hipStream_t();
        hipError_t status = hipStreamCreate( *pStream );
        if (status != hipSuccess) throw std::runtime_error("hipStreamCreate failed!");
    }

    // destroy an existing stream
    static void destroy( hipStream_t *pStream ){
        hipError_t status = hipStreamDestroy(*pStream);
        if (status != hipSuccess) throw std::runtime_error("hipStreamDestroy failed!");
        delete pStream;
        pStream = nullptr;
    }

    // synchronize an existing stream
    static void sync( hipStream_t *pStream ){
        hipError_t status = hipStreamSynchronize(*pStream);
        if (status != hipSuccess) throw std::runtime_error("hipStreamSynchronize failed!");
    }
};

} // namespace ppt

#endif