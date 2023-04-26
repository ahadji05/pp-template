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
#ifndef PPT_MEM_SPACE_HOST_HPP
#define PPT_MEM_SPACE_HOST_HPP

#include <cstdlib>
#include <cstring>

#include "ppt/memory/MemorySpace.hpp"

namespace ppt
{

/**
 * @brief A class that represents MemorySpace for Host calls!
 * Implements the following operations: allocate, release, copy,
 * copyToHost, copyFromHost.
 * Build on top of STD library.
 *
 */
class MemSpaceHost : public MemorySpaceBase
{
  public:
    /**
     * @brief Static member class for memory allocations on Host.
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
        std::cout << "MemSpaceHost: allocator" << std::endl;
#endif

        *ptr = new value_t[n_elems];
        if (*ptr == nullptr) return message::allocation_failed;

        return message::no_error;
    }

    /**
     * @brief Static member class for memory freeing on Host.
     *
     * @tparam value_t Type of the elements to be deallocated.
     * @param ptr Pointer to array to free.
     * @return return_t Returns release_failed=true if something went wrong with
     * memory release, else it returns false.
     */
    template <typename value_t> static return_t release(value_t *ptr)
    {
#ifdef PPT_DEBUG_MEMORY_MANAGE
        std::cout << "MemSpaceHost: release" << std::endl;
#endif
        try
        {
            delete[] ptr;
        }

        catch (const std::exception &e)
        {
            return message::release_failed;
        }

        return message::no_error;
    }

    /**
     * @brief Static member class for copying data within this memory space.
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
        std::cout << "MemSpaceHost: copy" << std::endl;
#endif

        length_t n_bytes = n_elems * sizeof(value_t);
        try
        {
            std::memcpy(to, from, n_bytes);
        }

        catch (const std::exception &e)
        {
            return message::copying_failed;
        }

        return message::no_error;
    }

    /**
     * @brief Static member class for copying data To Host. (In practice, since
     * this is the Host memory space this is simply the same operation as the
     * previous copy routine).
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
        std::cout << "MemSpaceHost: copyToHost" << std::endl;
#endif

        length_t n_bytes = n_elems * sizeof(value_t);
        try
        {
            std::memcpy(to, from, n_bytes);
        }

        catch (const std::exception &e)
        {
            return message::copying_failed;
        }

        return message::no_error;
    }

    /**
     * @brief Static member class for copying data From Host. (In practice, since
     * this is the Host memory space this is simply the same operation as the
     * previous copy routine).
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
        std::cout << "MemSpaceHost: copyFromHost" << std::endl;
#endif

        length_t n_bytes = n_elems * sizeof(value_t);
        try
        {
            std::memcpy(to, from, n_bytes);
        }

        catch (const std::exception &e)
        {
            return message::copying_failed;
        }

        return message::no_error;
    }
};

} // namespace ppt

#endif // PPT_MEM_SPACE_HOST_HPP
