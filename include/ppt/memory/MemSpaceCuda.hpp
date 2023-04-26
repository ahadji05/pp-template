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
#ifndef PPT_MEM_SPACE_CUDA_HPP
#define PPT_MEM_SPACE_CUDA_HPP

#include "ppt/memory/MemorySpace.hpp"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

namespace ppt
{

/**
 * @brief A class that represents MemorySpace for Cuda device calls!
 * Implements the following operations: allocate, release, copy,
 * copyToHost, copyFromHost.
 * Build on top of CUDA run-time API.
 *
 */
class MemSpaceCuda : public MemorySpaceBase
{
  public:
    /**
     * @brief Static member class for memory allocations on Cuda device.
     *
     * @tparam value_t Type of the elements to be allocated.
     * @tparam length_t Type of number of elements. Must be an integral type.
     * @param ptr Pointer of type=value_t, to the allocated array.
     * @param n_elems Number of elements.
     * @return return_t Returns allocation_failed=true if something went wrong
     * with memory allocation, else it returns false.
     */
    template <typename value_t, typename length_t>
    static typename std::enable_if<std::is_integral<length_t>::value, return_t>::type allocate(value_t **ptr,
                                                                                               length_t n_elems)
    {
#ifdef PPT_DEBUG_MEMORY_MANAGE
        std::cout << "MemSpaceCuda: allocator" << std::endl;
#endif

        length_t n_bytes   = n_elems * sizeof(value_t);
        cudaError_t status = cudaMalloc((void **)ptr, (size_t)n_bytes);
        if (status != cudaSuccess) return message::allocation_failed;

        return message::no_error;
    }

    /**
     * @brief Static member class for memory freeing on Cuda Device.
     *
     * @tparam value_t Type of the elements to be deallocated.
     * @param ptr Pointer to array to free.
     * @return return_t Returns release_failed=true if something went wrong with
     * memory release, else it returns false.
     */
    template <typename value_t> static return_t release(value_t *ptr)
    {
#ifdef PPT_DEBUG_MEMORY_MANAGE
        std::cout << "MemSpaceCuda: release" << std::endl;
#endif

        cudaError_t status = cudaFree((void *)ptr);
        if (status != cudaSuccess) return message::release_failed;

        return message::no_error;
    }

    /**
     * @brief Static member class for copying data within Cuda device memory.
     *
     * @tparam value_t Type of the elements to be copied.
     * @tparam length_t Type of number of elements. Must be an integral type.
     * @param to Destination memory address.
     * @param from Source memory address.
     * @param n_elems Number of elements to be copied.
     * @return return_t Returns copying_failed=true if something went wrong with
     * memory copy, else it returns false.
     */
    template <typename value_t, typename length_t>
    static typename std::enable_if<std::is_integral<length_t>::value, return_t>::type copy(value_t *to, value_t *from,
                                                                                           length_t n_elems)
    {
#ifdef PPT_DEBUG_MEMORY_MANAGE
        std::cout << "MemSpaceCuda: copy" << std::endl;
#endif

        length_t n_bytes   = n_elems * sizeof(value_t);
        cudaError_t status = cudaMemcpy((void *)to, (void *)from, (size_t)n_bytes, cudaMemcpyDeviceToDevice);
        if (status != cudaSuccess) return message::copying_failed;

        return message::no_error;
    }

    /**
     * @brief Static member class for copying data from Cuda device memory to
     * Host memory.
     *
     * @tparam value_t Type of the elements to be copied.
     * @tparam length_t Type of number of elements. Must be an integral type.
     * @param to Destination memory address.
     * @param from Source memory address.
     * @param n_elems Number of elements to be copied.
     * @return return_t Returns copying_failed=true if something went wrong with
     * memory copy, else it returns false.
     */
    template <typename value_t, typename length_t>
    static typename std::enable_if<std::is_integral<length_t>::value, return_t>::type copyToHost(value_t *to,
                                                                                                 value_t *from,
                                                                                                 length_t n_elems)
    {
#ifdef PPT_DEBUG_MEMORY_MANAGE
        std::cout << "MemSpaceCuda: copyToHost" << std::endl;
#endif

        length_t n_bytes   = n_elems * sizeof(value_t);
        cudaError_t status = cudaMemcpy((void *)to, (void *)from, (size_t)n_bytes, cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) return message::copying_failed;

        return message::no_error;
    }

    /**
     * @brief Static member class for copying data to Cuda device memory from
     * Host memory.
     *
     * @tparam value_t Type of the elements to be copied.
     * @tparam length_t Type of number of elements. Must be an integral type.
     * @param to Destination memory address.
     * @param from Source memory address.
     * @param n_elems Number of elements to be copied.
     * @return return_t Returns copying_failed=true if something went wrong with
     * memory copy, else it returns false.
     */
    template <typename value_t, typename length_t>
    static typename std::enable_if<std::is_integral<length_t>::value, return_t>::type copyFromHost(value_t *to,
                                                                                                   value_t *from,
                                                                                                   length_t n_elems)
    {
#ifdef PPT_DEBUG_MEMORY_MANAGE
        std::cout << "MemSpaceCuda: copyFromHost" << std::endl;
#endif

        length_t n_bytes   = n_elems * sizeof(value_t);
        cudaError_t status = cudaMemcpy((void *)to, (void *)from, (size_t)n_bytes, cudaMemcpyHostToDevice);
        if (status != cudaSuccess) return message::copying_failed;

        return message::no_error;
    }
};

} // namespace ppt

#endif // PPT_MEM_SPACE_CUDA_HPP
