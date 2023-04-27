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
#ifndef SCALAR_FIELD_HPP
#define SCALAR_FIELD_HPP

#include "ppt/containers/Vector.hpp"
#include "ppt/memory/MemorySpacesInc.hpp"
#include "ppt/types.hpp"

/**
 * @brief
 *
 * @tparam MemSpace is the Memory Space that handles the memory allocations,
 * copies, and release operations.
 */
template <class MemSpace> class ScalarField
{
    // The container can only be instantiated using a valid MemorySpace provide by
    // "memory/MemorySpacesInc.hpp"
    static_assert(ppt::is_memory_space<MemSpace>::value,
                  "ScalarField: The provided class MemSpace in not a valid MemorySpace.");

  public:
    // Alias the vector-type using the float_type defined in "types.hpp", and the
    // provided template parameter MemSpace.
    using vector_type = ppt::Vector<float_type, MemSpace>;

    ScalarField();
    ScalarField(size_t nz, size_t nx);
    ScalarField(size_t nz, size_t nx, vector_type &othervector);
    ScalarField(const ScalarField &otherScalarField);
    ScalarField &operator=(const ScalarField &otherScalarField);
    ~ScalarField();

  protected:
    vector_type _field;            // vector that contains the data in MemSpace
    size_t _nz, _nx;               // dimensions
    bool dimsMatchnElems() const;  // assert that: _nz * _nx == _field._nElems
    vector_type get_field() const; // get _field using deep-copy

  public:
    inline float_type *get_ptr() const;
    inline size_t get_nz() const;
    inline size_t get_nx() const;
    inline size_t get_nElems() const;
    void swap(ScalarField &other);
};

template <class MemSpace> ScalarField<MemSpace>::ScalarField()
{
    _nz = 0;
    _nx = 0;
    // _field is constructed using it's own default constructor
}

template <class MemSpace> ScalarField<MemSpace>::ScalarField(size_t nz, size_t nx)
{
    _nz    = nz;
    _nx    = nx;
    _field = vector_type(nz * nx);
    assert(this->dimsMatchnElems());
}

template <class MemSpace> ScalarField<MemSpace>::ScalarField(size_t nz, size_t nx, vector_type &othervector)
{
    _nz    = nz;
    _nx    = nx;
    _field = othervector;
    assert(this->dimsMatchnElems());
}

template <class MemSpace> ScalarField<MemSpace>::ScalarField(const ScalarField &otherScalarField)
{
    _nz    = otherScalarField.get_nz();
    _nx    = otherScalarField.get_nx();
    _field = otherScalarField.get_field();
    assert(this->dimsMatchnElems());
}

template <class MemSpace> ScalarField<MemSpace> &ScalarField<MemSpace>::operator=(const ScalarField &otherScalarField)
{
    if (this != &otherScalarField)
    {
        _nz = otherScalarField.get_nz();
        _nx = otherScalarField.get_nx();

        if (this->_field.get_nElems() == otherScalarField._field.get_nElems())
        {
            /* reuse existing memory and avoid copy-assign that involves: release,
             * allocation, and copy.*/
            MemSpace::copy(_field.get_ptr(), otherScalarField.get_ptr(), _field.get_nElems());
        }
        else
        {
            // deep copy using the copy-assignment operator=
            _field = otherScalarField.get_field();
        }
    }
    assert(this->dimsMatchnElems());

    return *this;
}

template <class MemSpace> ScalarField<MemSpace>::~ScalarField()
{
    _nz = 0;
    _nx = 0;
    // _field is implicitly destroyed by it's own default destructor
}

template <class MemSpace> bool ScalarField<MemSpace>::dimsMatchnElems() const
{
    return (_nz * _nx == _field.get_nElems());
}

template <class MemSpace> typename ScalarField<MemSpace>::vector_type ScalarField<MemSpace>::get_field() const
{
    return _field;
}

template <class MemSpace> inline float_type *ScalarField<MemSpace>::get_ptr() const
{
    return _field.get_ptr();
}

template <class MemSpace> inline size_t ScalarField<MemSpace>::get_nz() const
{
    return _nz;
}

template <class MemSpace> inline size_t ScalarField<MemSpace>::get_nx() const
{
    return _nx;
}

template <class MemSpace> inline size_t ScalarField<MemSpace>::get_nElems() const
{
    return _field.get_nElems();
}

template <class MemSpace> void ScalarField<MemSpace>::swap(ScalarField &other)
{
    std::swap(_nz, other._nz);
    std::swap(_nx, other._nx);
    _field.swap(other._field);
}

#endif