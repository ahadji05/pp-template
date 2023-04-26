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
#ifndef PPT_TIME_EXTRAP_HPP
#define PPT_TIME_EXTRAP_HPP

#include "ppt/containers/ScalarField.hpp"
#include "ppt/execution/ExecutionSpacesInc.hpp"
#include "ppt/types.hpp"

/**
 * @brief This routine performs wavefield time extrapolation using the FD
 * scheme:
 *
 *      pnew = (2*p - pold) + (dt/dh)^2 * v(z,x)^2 * (pzz + pxx)
 *
 * @tparam ExecSpace Execution-Space that is used to resolve the back-end
 * implementation at compile-time.
 * @tparam MemSpace Memory-Space that must be accessible from the
 * Execution-Space; otherwise compile-time error.
 * @param pnew wavefield for the next time-step
 * @param p wavefield one time-step old
 * @param pold wavefield two time-steps old
 * @param pxx second-order derivative of the wavefield (p) in the x dimension
 * @param pxx second-order derivative of the wavefield (p) in the z dimension
 * @param velmodel velocity profile that can be space-dependent
 * @param dt time-step in seconds
 * @param dh space-step in meters
 * @param tag tag for dispatching the selected Execution-Space
 * @return void IF-ONLY the Memory-Space is accessible from the Execution-Space;
 * otherwise it produces compile-time error.
 */
template <class ExecSpace, class MemSpace>
typename std::enable_if<std::is_same<MemSpace, typename ExecSpace::accessible_space>::value, void>::type fd_time_extrap(
    ScalarField<MemSpace> &pnew, const ScalarField<MemSpace> &p, const ScalarField<MemSpace> &pold,
    const ScalarField<MemSpace> &pxx, const ScalarField<MemSpace> &pzz, const ScalarField<MemSpace> &velmodel,
    float_type dt, float_type dh, ExecSpace tag);

#endif
