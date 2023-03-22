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
#ifndef TMP_VECTOR_HPP
#define TMP_VECTOR_HPP

#include "memory/MemorySpacesInc.hpp"

namespace TMP
{

/**
 * @brief Vector is a basic container for storing data in a contiguous memory
 * segment. The template parameter <memory_space> enables managing data in
 * different Memory Spaces, depending on the provided type. The Vector
 * container provides a public interface of methods for accessing and managing
 * its member data.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam length_t Type of elements count. Must be an integral type (int,
 * size_t, etc.).
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 */
template <typename value_t, typename length_t, class memory_space> class Vector
{

    static_assert(TMP::is_memory_space<memory_space>::value, "Vector: The provided class in not a Memory Space.");
    static_assert(std::is_integral<length_t>::value, "Vector: The provide class length_t is not integral type.");

  public:
    Vector();
    Vector(length_t nElems);
    Vector(value_t *p, length_t nElems);
    Vector(const Vector &other_vector);
    Vector &operator=(const Vector &other_vector);
    ~Vector();

  protected:
    length_t _nElems;
    length_t _nBytes;
    value_t *_ptr;

  public:
    inline length_t get_nElems() const;
    inline length_t get_nBytes() const;
    inline value_t *get_ptr();
    inline value_t &operator[](length_t i);
};

/**
 * @brief Construct a new Vector<value_t, length_t, memory_space>::Vector
 * object.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam length_t Type of elements count. Must be an integral type (int,
 * size_t, etc.).
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 */
template <typename value_t, typename length_t, class memory_space> Vector<value_t, length_t, memory_space>::Vector()
{
    _nElems = 0;
    _nBytes = 0;
    _ptr = nullptr;
}

/**
 * @brief Construct a new Vector<value_t, length_t, memory_space>::Vector
 * object.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam length_t Type of elements count. Must be an integral type (int,
 * size_t, etc.).
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 * @param nElems Number of elements that fit in the Vector.
 */
template <typename value_t, typename length_t, class memory_space>
Vector<value_t, length_t, memory_space>::Vector(length_t nElems)
{
    _nElems = nElems;
    _nBytes = nElems * sizeof(value_t);
    memory_space::allocate(&_ptr, _nBytes);
}

/**
 * @brief Construct a new Vector<value_t, length_t, memory_space>::Vector
 * object.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam length_t Type of elements count. Must be an integral type (int,
 * size_t, etc.).
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 * @param p Pointer from where data are copied into this Vector.
 * @param nElems Number of elements that fit in the Vector.
 */
template <typename value_t, typename length_t, class memory_space>
Vector<value_t, length_t, memory_space>::Vector(value_t *p, length_t nElems)
{
    _nElems = nElems;
    _nBytes = nElems * sizeof(value_t);
    memory_space::allocate(&_ptr, _nBytes);
    memory_space::copy(_ptr, p, _nBytes);
}

/**
 * @brief Construct a new Vector<value_t, length_t, memory_space>::Vector
 * object.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam length_t Type of elements count. Must be an integral type (int,
 * size_t, etc.).
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 * @param other_vector Another Vector from which this is Constructed.
 */
template <typename value_t, typename length_t, class memory_space>
Vector<value_t, length_t, memory_space>::Vector(const Vector &other_vector)
{
    _nElems = other_vector._nElems;
    _nBytes = other_vector._nBytes;
    memory_space::allocate(&_ptr, _nBytes);
    memory_space::copy(_ptr, other_vector._ptr, _nBytes);
}

/**
 * @brief Copy Vector.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam length_t Type of elements count. Must be an integral type (int,
 * size_t, etc.).
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 * @param other_vector Another Vector which is copied into this.
 * @return Vector<value_t, length_t, memory_space>&
 */
template <typename value_t, typename length_t, class memory_space>
Vector<value_t, length_t, memory_space> &Vector<value_t, length_t, memory_space>::operator=(const Vector &other_vector)
{
    if (this != &other_vector)
    {
        if (this->_nBytes != other_vector._nBytes) /* cannot reuse memory */
        {
            _nElems = other_vector._nElems;
            _nBytes = other_vector._nBytes;
            memory_space::release(_ptr);
            memory_space::allocate(&_ptr, _nBytes);
        }
        memory_space::copy(_ptr, other_vector._ptr, _nBytes);
    }
    return *this;
}

/**
 * @brief Destroy the Vector<value_t, length_t, memory_space>::Vector object.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam length_t Type of elements count. Must be an integral type (int,
 * size_t, etc.).
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 */
template <typename value_t, typename length_t, class memory_space> Vector<value_t, length_t, memory_space>::~Vector()
{
    memory_space::release(_ptr);
    _ptr = nullptr;
}

/**
 * @brief Get number of bytes.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam length_t Type of elements count. Must be an integral type (int,
 * size_t, etc.).
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 * @return length_t
 */
template <typename value_t, typename length_t, class memory_space>
inline length_t Vector<value_t, length_t, memory_space>::get_nBytes() const
{
    return _nBytes;
}

/**
 * @brief Get number of elements.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam length_t Type of elements count. Must be an integral type (int,
 * size_t, etc.).
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 * @return length_t
 */
template <typename value_t, typename length_t, class memory_space>
inline length_t Vector<value_t, length_t, memory_space>::get_nElems() const
{
    return _nElems;
}

/**
 * @brief Get the memory address of _ptr.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam length_t Type of elements count. Must be an integral type (int,
 * size_t, etc.).
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 * @return value_t*
 */
template <typename value_t, typename length_t, class memory_space>
inline value_t *Vector<value_t, length_t, memory_space>::get_ptr()
{
    return _ptr;
}

/**
 * @brief Get an element from the Vector based on index i.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam length_t Type of elements count. Must be an integral type (int,
 * size_t, etc.).
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 * @param i Element index.
 * @return value_t&
 */
template <typename value_t, typename length_t, class memory_space>
inline value_t &Vector<value_t, length_t, memory_space>::operator[](length_t i)
{
    static_assert(std::is_same<memory_space, TMP::MemSpaceHost>::value,
                  "Vector operator[]: Cannot access non-host elements.");

#ifdef TMP_ENABLE_BOUND_CHECK
    assert(i <= _nElems);
#endif

    return _ptr[i];
}

} // namespace TMP

#endif // TMP_VECTOR_HPP