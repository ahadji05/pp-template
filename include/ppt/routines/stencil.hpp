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
#ifndef PPT_STENCIL_HPP
#define PPT_STENCIL_HPP

#include "ppt/containers/ScalarField.hpp"
#include "ppt/execution/ExecutionSpacesInc.hpp"
#include "ppt/types.hpp"

/**
 * @brief This routine calculates the second-order derivative pxx of a wavefield
 * p using the FD scheme:
 *
 *  pxx[z,x] = -1/12*p[z,x-2] +4/3*p[z,x-1] -5/2*p[z,x] +4/3*p[z,x+1]
 * -1/12*p[z,x+2]
 *
 * @tparam ExecSpace Execution-Space that is used to resolve the back-end
 * implementation at compile-time.
 * @tparam MemSpace Memory-Space that must be accessible from the
 * Execution-Space; otherwise compile-time error.
 * @param pxx wavefield's second-order derivative in x dimension
 * @param p wavefield from which the derivative pxx is calculated
 * @param tag tag for dispatching the selection Execution-Space
 * @return void IF-ONLY the Memory-Space is accessible from the Execution-Space;
 * otherwise it produces compile-time error.
 */
template <class ExecSpace, class MemSpace>
typename std::enable_if<std::is_same<typename ExecSpace::accessible_space, MemSpace>::value, void>::type fd_pxx(
    ScalarField<MemSpace> &pxx, const ScalarField<MemSpace> &p, ExecSpace tag);

/**
 * @brief This routine calculates the second-order derivative pzz of a wavefield
 * p using the FD scheme:
 *
 *  pzz[z,x] = -1/12*p[z-2,x] +4/3*p[z-1,x] -5/2*p[z,x] +4/3*p[z+1,x]
 * -1/12*p[z+2,x]
 *
 * @tparam ExecSpace Execution-Space that is used to resolve the back-end
 * implementation at compile-time.
 * @tparam MemSpace Memory-Space that must be accessible from the
 * Execution-Space; otherwise compile-time error.
 * @param pzz wavefield's second-order derivative in z dimension
 * @param p wavefield from which the derivative pzz is calculated
 * @param tag tag for dispatching the selection Execution-Space
 * @return void IF-ONLY the Memory-Space is accessible from the Execution-Space;
 * otherwise it produces compile-time error.
 */
template <class ExecSpace, class MemSpace>
typename std::enable_if<std::is_same<typename ExecSpace::accessible_space, MemSpace>::value>::type fd_pzz(
    ScalarField<MemSpace> &pzz, const ScalarField<MemSpace> &p, ExecSpace tag);

#endif
