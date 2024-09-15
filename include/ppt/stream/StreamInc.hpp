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
#ifndef PPT_STREAM_INC_HPP
#define PPT_STREAM_INC_HPP

#include "ppt/definitions.hpp"
#include "ppt/stream/StreamHost.hpp"

#ifdef PPT_ENABLE_CUDA_BACKEND
#include "ppt/stream/StreamCuda.hpp"
#endif

#ifdef PPT_ENABLE_HIP_BACKEND
#include "ppt/stream/StreamHip.hpp"
#endif

namespace ppt
{
/**
 * @brief Assert that a given type T is a Stream. A valid
 * Stream type is one that is derived from the class StreamBase.
 *
 * @tparam T The type to check if it is derived from class StreamBase.
 */
template <typename T> struct is_stream_t
{
    static constexpr bool value = std::is_base_of<StreamBase,T>::value;
};

} // namespace ppt

#endif