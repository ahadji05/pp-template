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
 * @param pnew wavefield for the next time-step
 * @param p wavefield one time-step old
 * @param pold wavefield two time-steps old
 * @param pxx second-order derivative of the wavefield (p) in the x dimension
 * @param pxx second-order derivative of the wavefield (p) in the z dimension
 * @param velmodel velocity profile that can be space-dependent
 * @param dt time-step in seconds
 * @param dh space-step in meters
 * @param stream to use for the execution
 * @param tag for dispatching the selected Execution-Space
 */
template <class ExecSpace>
void fd_time_extrap(
    ScalarField<typename ExecSpace::accessible_space> &pnew, 
    const ScalarField<typename ExecSpace::accessible_space> &p, 
    const ScalarField<typename ExecSpace::accessible_space> &pold,
    const ScalarField<typename ExecSpace::accessible_space> &pxx, 
    const ScalarField<typename ExecSpace::accessible_space> &pzz, 
    const ScalarField<typename ExecSpace::accessible_space> &velmodel,
    float_type dt, 
    float_type dh, 
    [[maybe_unused]] typename ExecSpace::stream_space::type stream, 
    ExecSpace tag
);

#endif
