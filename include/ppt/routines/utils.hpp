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
#ifndef PPT_UTILS_HPP
#define PPT_UTILS_HPP

#include "ppt/execution/ExecutionSpacesInc.hpp"
#include "ppt/containers/Vector.hpp"
#include "ppt/types.hpp"

/**
 * @brief This routine sets at a specified position (iz,ix) of a specified
 * wavefield (p) the source-amplitude (src).
 *
 * @tparam ExecSpace Execution-Space that is used to resolve the back-end implementation at compile-time.
 * @param p wavefield
 * @param vec vector to apply scaling over
 * @param scaling value to multiply each element of the vector with
 * @param stream to use for the execution
 * @param tag for dispatching the selection Execution-Space
 */
template <class ExecSpace>
void scale(
    int nData,
    float_type *data, 
    float_type scale, 
    [[maybe_unused]] typename ExecSpace::stream_space::type stream, 
    ExecSpace tag
);

#endif