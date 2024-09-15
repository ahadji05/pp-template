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
#ifndef PPT_STREAM_HOST
#define PPT_STREAM_HOST

#include "ppt/stream/StreamBase.hpp"

namespace ppt 
{

struct StreamHostType{};

class StreamHost : public StreamBase {

  public:
    using type = StreamHostType;

    // create a new stream
    static void create( StreamHostType **pStream ) {
        *pStream = new StreamHostType();
    }

    // destroy an existing stream
    static void destroy( StreamHostType *pStream ){
        delete pStream;
    }

    // sync an existing stream
    static void sync( StreamHostType *pStream ){
        (void)pStream; // intentionally unused!
    }
};

} // namespace ppt

#endif