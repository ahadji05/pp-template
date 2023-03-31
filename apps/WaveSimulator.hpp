
#include "containers/ScalarField.hpp"
#include "routines/add_source.hpp"
#include "routines/stencil.hpp"
#include "routines/time_extrap.hpp"

#include <fstream>
#include <vector>

/**
 * @brief This class implements wavefield simulation based on the 2D
 * finite-difference scheme:
 *      Pnew = (2*P - Pold) + (dt/dh)^2 * V(z,x)^2 * (Pzz + Pxx)
 *
 * @note dt: is the time step in seconds,
 * @note dh: is the space step in meters (dz=dx=dh),
 * @note V(z,x): is the velocity profile that may be inhomogeneous
 *
 * @tparam ExecSpace The execution-space resolves the accessible
 * memory-space and compiles the code based on the appropriate
 * containers and routines.
 */
template <class ExecSpace> class WaveSimulator
{
  private:
    using MemSpace = typename ExecSpace::accessible_space;

    ScalarField<MemSpace> wavefield;
    ScalarField<MemSpace> wavefield_new;
    ScalarField<MemSpace> wavefield_old;
    ScalarField<MemSpace> wavefield_pxx;
    ScalarField<MemSpace> wavefield_pzz;
    ScalarField<MemSpace> velmodel;
    float_type _dt, _dh;
    float_type _vmin;
    size_t _srcz, _srcx;
    size_t _nt, _nz, _nx;
    std::vector<float_type> source_impulse;

  public:
    WaveSimulator() = default;
    ~WaveSimulator() = default;

    void set_time_step(float_type dt);
    void set_space_step(float_type dt);
    void set_number_of_time_steps(size_t nt);
    void set_source_position_x(size_t ix);
    void set_source_position_z(size_t iz);
    void set_dimensions(size_t nz, size_t nx);
    void set_vmin(float_type vmin);
    void set_velocity_layer(size_t izmin, size_t izmax, float_type v);

    void make_ricker(float_type fpeak);
    void store_velmodel_to_binary(const char *filename) const;
    void store_wavefield_to_binary(const char *filename) const;

    float_type CLF_condition() const;

    void run()
    {
        for (size_t i(0); i < _nt; ++i)
        {
            if (i % 250 == 0)
                std::cout << "time-step: " << i << std::endl;

            add_source(wavefield, source_impulse[i], _srcx, _srcz, ExecSpace());

            fd_pxx(wavefield_pxx, wavefield, ExecSpace());

            fd_pzz(wavefield_pzz, wavefield, ExecSpace());

            fd_time_extrap(wavefield_new, wavefield, wavefield_old, wavefield_pxx, wavefield_pzz, velmodel, _dt, _dh,
                           ExecSpace());

            wavefield_old = wavefield;
            wavefield = wavefield_new;
        }
    }
};

template <class ExecSpace> void WaveSimulator<ExecSpace>::set_time_step(float_type dt)
{
    _dt = dt;
}

template <class ExecSpace> void WaveSimulator<ExecSpace>::set_space_step(float_type dh)
{
    _dh = dh;
}

template <class ExecSpace> void WaveSimulator<ExecSpace>::set_number_of_time_steps(size_t nt)
{
    _nt = nt;
    source_impulse.resize(_nt);
}

template <class ExecSpace> void WaveSimulator<ExecSpace>::set_source_position_z(size_t iz)
{
    _srcz = iz;
}

template <class ExecSpace> void WaveSimulator<ExecSpace>::set_source_position_x(size_t ix)
{
    _srcx = ix;
}

template <class ExecSpace> void WaveSimulator<ExecSpace>::set_dimensions(size_t nz, size_t nx)
{
    _nz = nz;
    _nx = nx;

    // construct all fields based on the specified dimension
    this->wavefield = ScalarField<MemSpace>(_nz, _nx);
    this->wavefield_new = ScalarField<MemSpace>(_nz, _nx);
    this->wavefield_old = ScalarField<MemSpace>(_nz, _nx);
    this->wavefield_pxx = ScalarField<MemSpace>(_nz, _nx);
    this->wavefield_pzz = ScalarField<MemSpace>(_nz, _nx);
    this->velmodel = ScalarField<MemSpace>(_nz, _nx);
}

template <class ExecSpace> void WaveSimulator<ExecSpace>::set_vmin(float_type vmin)
{
    _vmin = vmin;
    // allocate array on Host
    float_type *data_host;
    TMP::MemSpaceHost::allocate(&data_host, velmodel.get_nElems());

    // copy the velocity profile from MemSpace to host-array
    MemSpace::copyToHost(data_host, velmodel.get_ptr(), velmodel.get_nElems());

    // make all values that are smaller than vmin equal to it
    for (size_t i(0); i < velmodel.get_nElems(); ++i)
        if (data_host[i] < vmin)
            data_host[i] = vmin;

    // copy data to MemSpace and then deallocate the host array
    MemSpace::copyFromHost(velmodel.get_ptr(), data_host, velmodel.get_nElems());
    TMP::MemSpaceHost::release(data_host);
}

template <class ExecSpace> void WaveSimulator<ExecSpace>::make_ricker(float_type fpeak)
{
    // allocate array on Host
    float_type *data_host;
    TMP::MemSpaceHost::allocate(&data_host, _nt);

    // compute half Ricker-wavelet on host array
    data_host[0] = 1.0;
    for (int it = 1; it < _nt; ++it)
    {
        float_type t = it * _dt;
        float_type term = M_PI * M_PI * fpeak * fpeak * t * t;
        float_type amplitude = (1 - 2 * term) * exp(-term);
        data_host[it] = amplitude;
    }

    // copy data to MemSpace and then deallocate the host array
    MemSpace::copyFromHost(source_impulse.data(), data_host, _nt);
    TMP::MemSpaceHost::release(data_host);
}

template <class ExecSpace> void WaveSimulator<ExecSpace>::set_velocity_layer(size_t izmin, size_t izmax, float_type v)
{
    // assert validity of the zmin and zmax ranges
    assert(izmax <= _nz);
    assert(izmin <= _nz && izmin <= izmax);

    // allocate array on host
    float_type *data_host;
    size_t nelems = (izmax - izmin) * _nx;
    TMP::MemSpaceHost::allocate(&data_host, nelems);

    // set the specified velocity value
    size_t offset = izmin * _nx;
    for (size_t i(0); i < nelems; ++i)
        data_host[i] = v;

    // calculate offset on array on MemSpace, and then copy the data
    float_type *offset_mem_space_ptr = velmodel.get_ptr() + offset;
    MemSpace::copyFromHost(offset_mem_space_ptr, data_host, nelems);

    // deallocate the host-array
    TMP::MemSpaceHost::release(data_host);
}

template <class ExecSpace> void WaveSimulator<ExecSpace>::store_velmodel_to_binary(const char *filename) const
{
    std::fstream file;
    file.open(filename, std::ios_base::out | std::ios_base::binary);

    if (!file.is_open())
        throw std::runtime_error("Unable to open file!");

    // allocate array on host, copy the data into, and then store to binary file
    float_type *data_host;
    TMP::MemSpaceHost::allocate(&data_host, velmodel.get_nElems());
    MemSpace::copyToHost(data_host, velmodel.get_ptr(), velmodel.get_nElems());

    // write data to binary file
    file.write((char *)data_host, sizeof(float) * velmodel.get_nElems());

    // deallocate the host-array and close file
    TMP::MemSpaceHost::release(data_host);
    file.close();
}

template <class ExecSpace> void WaveSimulator<ExecSpace>::store_wavefield_to_binary(const char *filename) const
{
    std::fstream file;
    file.open(filename, std::ios_base::out | std::ios_base::binary);

    if (!file.is_open())
        throw std::runtime_error("Unable to open file!");

    // allocate array on host, copy the data into, and then store to binary file
    float_type *data_host;
    TMP::MemSpaceHost::allocate(&data_host, wavefield.get_nElems());
    MemSpace::copyToHost(data_host, wavefield.get_ptr(), wavefield.get_nElems());

    // write data to binary file
    file.write((char *)data_host, sizeof(float) * wavefield.get_nElems());

    // deallocate the host-array and close file
    TMP::MemSpaceHost::release(data_host);
    file.close();
}

template <class ExecSpace> float_type WaveSimulator<ExecSpace>::CLF_condition() const
{
    return _vmin * _dt / _dh;
}