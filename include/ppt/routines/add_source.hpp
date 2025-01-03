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
#ifndef PPT_ADD_SOURCE_HPP
#define PPT_ADD_SOURCE_HPP

#include "ppt/containers/ScalarField.hpp"
#include "ppt/execution/ExecutionSpacesInc.hpp"
#include "ppt/types.hpp"

/**
 * @brief This routine sets at a specified position (iz,ix) of a specified
 * wavefield (p) the source-amplitude (src).
 *
 * @tparam ExecSpace Execution-Space that is used to resolve the back-end implementation at compile-time.
 * @param p wavefield
 * @param src source amplitude at given location
 * @param ix source index across the x dimension
 * @param iz source index across the z dimension
 * @param stream to use for the execution
 * @param tag for dispatching the selection Execution-Space
 */
template <class ExecSpace>
void add_source( 
    ScalarField<typename ExecSpace::accessible_space> &p, 
    const float_type src, 
    size_t ix, 
    size_t iz, 
    [[maybe_unused]] typename ExecSpace::stream_space::type stream, 
    ExecSpace tag
);

#endif
