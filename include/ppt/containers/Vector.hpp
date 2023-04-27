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
#ifndef PPT_VECTOR_HPP
#define PPT_VECTOR_HPP

#include "ppt/memory/MemorySpacesInc.hpp"

namespace ppt
{

/**
 * @brief Vector is a basic container for storing data in a contiguous memory
 * segment. The template parameter <memory_space> enables managing data in
 * different Memory Spaces, depending on the provided type. The Vector
 * container provides a public interface of methods for accessing and managing
 * its member data.
 *
 * @tparam value_t Type of values stored in the Vector.
 * size_t, etc.).
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 */
template <typename value_t, class memory_space> class Vector
{
    // Asset at compile-time that the specified memory_space is valid!
    static_assert(ppt::is_memory_space<memory_space>::value, "Vector: The provided class in not a Memory Space.");

  public:
    Vector();
    Vector(size_t nElems);
    Vector(value_t *p, size_t nElems);
    Vector(const Vector &other_vector);
    Vector &operator=(const Vector &other_vector);
    ~Vector();

  protected:
    size_t _nElems;
    value_t *_ptr;

  public:
    inline size_t get_nElems() const;
    inline value_t *get_ptr() const;
    inline value_t &operator[](size_t i) const;
    void swap(Vector &other_vector);

    void fill(value_t value);   // !Performance note: this method invokes
                                // Host-Device alloc/copy/free/sync, USE WISELY!
    void resize(size_t nElems); // !Performance note: this method invokes
                                // Host-Device alloc/copy/free/sync, USE WISELY!
};

/**
 * @brief Construct a new Vector<value_t, memory_space>::Vector object.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 */
template <typename value_t, class memory_space> Vector<value_t, memory_space>::Vector()
{
    _nElems = 0;
    _ptr    = nullptr;
}

/**
 * @brief Construct a new Vector<value_t, memory_space>::Vector
 * object.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 * @param nElems Number of elements that fit in the Vector.
 */
template <typename value_t, class memory_space> Vector<value_t, memory_space>::Vector(size_t nElems)
{
    _nElems = nElems;
    memory_space::allocate(&_ptr, _nElems);
    this->fill(0);
}

/**
 * @brief Construct a new Vector<value_t, memory_space>::Vector
 * object.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 * @param p Pointer from where data are copied into this Vector.
 * @param nElems Number of elements that fit in the Vector.
 */
template <typename value_t, class memory_space> Vector<value_t, memory_space>::Vector(value_t *p, size_t nElems)
{
    _nElems = nElems;
    memory_space::allocate(&_ptr, _nElems);
    memory_space::copyFromHost(_ptr, p, _nElems);
}

/**
 * @brief Construct a new Vector<value_t, memory_space>::Vector
 * object.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 * @param other_vector Another Vector from which this is Constructed.
 */
template <typename value_t, class memory_space> Vector<value_t, memory_space>::Vector(const Vector &other_vector)
{
    _nElems = other_vector._nElems;
    memory_space::allocate(&_ptr, _nElems);
    memory_space::copy(_ptr, other_vector._ptr, _nElems);
}

/**
 * @brief Copy Vector.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 * @param other_vector Another Vector which is copied into this.
 * @return Vector<value_t, memory_space>&
 */
template <typename value_t, class memory_space>
Vector<value_t, memory_space> &Vector<value_t, memory_space>::operator=(const Vector &other_vector)
{
    if (this != &other_vector)
    {
        if (this->_nElems != other_vector._nElems) /* cannot reuse memory */
        {
            _nElems = other_vector._nElems;
            memory_space::release(_ptr);
            memory_space::allocate(&_ptr, _nElems);
        }
        memory_space::copy(_ptr, other_vector._ptr, _nElems);
    }
    return *this;
}

/**
 * @brief Destroy the Vector<value_t, memory_space>::Vector object.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 */
template <typename value_t, class memory_space> Vector<value_t, memory_space>::~Vector()
{
    memory_space::release(_ptr);
    _ptr = nullptr;
}

/**
 * @brief Get number of elements.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 * @return size_t
 */
template <typename value_t, class memory_space> inline size_t Vector<value_t, memory_space>::get_nElems() const
{
    return _nElems;
}

/**
 * @brief Get the memory address of _ptr.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 * @return value_t*
 */
template <typename value_t, class memory_space> inline value_t *Vector<value_t, memory_space>::get_ptr() const
{
    return _ptr;
}

/**
 * @brief Get an element from the Vector based on index i.
 *
 * @tparam value_t Type of values stored in the Vector.
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 * @param i Element index.
 * @return value_t&
 */
template <typename value_t, class memory_space>
inline value_t &Vector<value_t, memory_space>::operator[](size_t i) const
{
    static_assert(std::is_same<memory_space, ppt::MemSpaceHost>::value,
                  "Vector operator[]: Cannot access non-host elements.");

#ifdef PPT_ENABLE_BOUND_CHECK
    assert(i <= _nElems);
#endif

    return _ptr[i];
}

/**
 * @brief Swap member variables between this and the incoming Vector.
 *
 * @tparam value_t Type of values/data stored in the Vector.
 * @tparam memory_space Memory Space that handles the memory allocations.
 */
template <typename value_t, class memory_space> void Vector<value_t, memory_space>::swap(Vector &other_vector)
{
    std::swap(_nElems, other_vector._nElems);
    std::swap(_ptr, other_vector._ptr);
}

/**
 * @brief Sets the value of each element in the Vector to the
 * specified one.
 *
 * @note This method invokes a Host::allocation/free, and a data
 * copy from Host to memory_space.
 *
 * @tparam value_t Type of values/data stored in the Vector.
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 * @param value The specified value to set each element in the Vector equal to.
 */
template <typename value_t, class memory_space> void Vector<value_t, memory_space>::fill(value_t value)
{
    value_t *data_host;
    ppt::MemSpaceHost::allocate(&data_host, _nElems);
    for (size_t i(0); i < _nElems; ++i)
        data_host[i] = value;
    memory_space::copyFromHost(_ptr, data_host, _nElems);
    ppt::MemSpaceHost::release(data_host);
}

/**
 * @brief Re-size the length of the Vector such that it containes
 * in total nElems.
 *
 * @note This method invokes a memory_space::allocation/free.
 * Then, by default, it fills the values to 0 (zero) using the method
 * this->fill(0).
 *
 * @tparam value_t Type of values/data stored in the Vector.
 * @tparam memory_space Memory Space that handles the memory allocations,
 * copies, and release operations.
 * @param nElems Number of elements to re-size Vector to.
 */
template <typename value_t, class memory_space> void Vector<value_t, memory_space>::resize(size_t nElems)
{
    if (nElems != this->_nElems)
    {
        memory_space::release(_ptr);
        _nElems = nElems;
        memory_space::allocate(&_ptr, _nElems);
        this->fill(0);
    }
}

} // namespace ppt

#endif // PPT_VECTOR_HPP
