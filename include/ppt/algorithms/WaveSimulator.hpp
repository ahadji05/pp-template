
#include <fstream>
#include <vector>

#include "ppt/containers/ScalarField.hpp"
#include "ppt/routines/add_source.hpp"
#include "ppt/routines/stencil.hpp"
#include "ppt/routines/time_extrap.hpp"

/**
 * @brief This class implements wavefield simulation based on the 2D
 * finite-difference scheme:
 *
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
template <class ExecSpace>
class WaveSimulator {
 private:
  using MemSpace = typename ExecSpace::accessible_space;

  ScalarField<MemSpace> wavefield;
  ScalarField<MemSpace> wavefield_new;
  ScalarField<MemSpace> wavefield_old;
  ScalarField<MemSpace> wavefield_pxx;
  ScalarField<MemSpace> wavefield_pzz;
  ScalarField<MemSpace> velmodel;

  std::vector<float_type> source_impulse;
  float_type _dt, _dh, _vmin, _vmax;
  size_t _nt, _nz, _nx, _srcz, _srcx;

 public:
  WaveSimulator()  = default;
  ~WaveSimulator() = default;

  // Set-methods
  void set_time_step(float_type dt);
  void set_space_step(float_type dt);
  void set_number_of_time_steps(size_t nt);
  void set_source_position_x(size_t ix);
  void set_source_position_z(size_t iz);
  void set_dimensions(size_t nz, size_t nx);
  void set_vmin(float_type vmin);
  void set_velocity_layer(size_t izmin, size_t izmax, float_type v);

  // Get-methods
  float_type get_time_step() const;
  float_type get_space_step() const;
  size_t get_number_of_time_steps() const;
  size_t get_source_position_x() const;
  size_t get_source_position_z() const;
  size_t get_dim_nx() const;
  size_t get_dim_nz() const;
  float_type get_vmin() const;
  float_type get_vmax() const;
  float_type get_CFL_condition() const;

  // Other public methods
  void make_ricker(float_type fpeak);
  void store_velmodel_to_binary(const char *filename) const;
  void store_wavefield_to_binary(const char *filename) const;
  void print_CFL_condition() const;

  // main algorithm
  void run() {
    for (size_t i(0); i < _nt; ++i) {
      if (i % 250 == 0) std::cout << "time-step: " << i << std::endl;

      add_source(wavefield, source_impulse[i], _srcx, _srcz, ExecSpace());

      fd_pxx(wavefield_pxx, wavefield, ExecSpace());

      fd_pzz(wavefield_pzz, wavefield, ExecSpace());

      fd_time_extrap(wavefield_new, wavefield, wavefield_old, wavefield_pxx,
                     wavefield_pzz, velmodel, _dt, _dh, ExecSpace());

      wavefield_old = wavefield;
      wavefield     = wavefield_new;
    }
  }
};

/**
 * @brief Set time step in seconds.
 *
 */
template <class ExecSpace>
void WaveSimulator<ExecSpace>::set_time_step(float_type dt) {
  _dt = dt;
}

/**
 * @brief Set space step in meters.
 *
 */
template <class ExecSpace>
void WaveSimulator<ExecSpace>::set_space_step(float_type dh) {
  _dh = dh;
}

/**
 * @brief Set the number of time extrapolation steps.
 *
 */
template <class ExecSpace>
void WaveSimulator<ExecSpace>::set_number_of_time_steps(size_t nt) {
  _nt = nt;
  source_impulse.resize(_nt);
}

/**
 * @brief Set source position index in the z dimension.
 *
 */
template <class ExecSpace>
void WaveSimulator<ExecSpace>::set_source_position_z(size_t iz) {
  _srcz = iz;
}

/**
 * @brief Set source position index in the x dimension.
 *
 */
template <class ExecSpace>
void WaveSimulator<ExecSpace>::set_source_position_x(size_t ix) {
  _srcx = ix;
}

/**
 * @brief Set the dimensions in z and x dimensions respectively.
 *
 */
template <class ExecSpace>
void WaveSimulator<ExecSpace>::set_dimensions(size_t nz, size_t nx) {
  _nz = nz;
  _nx = nx;

  // construct all fields based on the specified dimension
  this->wavefield     = ScalarField<MemSpace>(_nz, _nx);
  this->wavefield_new = ScalarField<MemSpace>(_nz, _nx);
  this->wavefield_old = ScalarField<MemSpace>(_nz, _nx);
  this->wavefield_pxx = ScalarField<MemSpace>(_nz, _nx);
  this->wavefield_pzz = ScalarField<MemSpace>(_nz, _nx);
  this->velmodel      = ScalarField<MemSpace>(_nz, _nx);
}

/**
 * @brief Set a minimum background velocity in the undelying model.
 *
 */
template <class ExecSpace>
void WaveSimulator<ExecSpace>::set_vmin(float_type vmin) {
  assert(vmin > 0);

  _vmin = vmin;
  // allocate array on Host
  float_type *data_host;
  TMP::MemSpaceHost::allocate(&data_host, velmodel.get_nElems());

  // copy the velocity profile from MemSpace to host-array
  MemSpace::copyToHost(data_host, velmodel.get_ptr(), velmodel.get_nElems());

  // make all values that are smaller than vmin equal to it
  for (size_t i(0); i < velmodel.get_nElems(); ++i)
    if (data_host[i] < vmin) data_host[i] = vmin;

  // copy data to MemSpace and then deallocate the host array
  MemSpace::copyFromHost(velmodel.get_ptr(), data_host, velmodel.get_nElems());
  TMP::MemSpaceHost::release(data_host);
}

/**
 * @brief Create a Ricker-wavelet:
 * R(t) = (1 - 2 pi^2 fpeak^2 t^2) * exp( pi^2 fpeak^2 t^2 )
 *
 */
template <class ExecSpace>
void WaveSimulator<ExecSpace>::make_ricker(float_type fpeak) {
  // allocate array on Host
  float_type *data_host;
  TMP::MemSpaceHost::allocate(&data_host, _nt);

  // compute half Ricker-wavelet on host array
  data_host[0] = 1.0;
  for (int it = 1; it < _nt; ++it) {
    float_type t         = it * _dt;
    float_type term      = M_PI * M_PI * fpeak * fpeak * t * t;
    float_type amplitude = (1 - 2 * term) * exp(-term);
    data_host[it]        = amplitude;
  }

  TMP::MemSpaceHost::copy(source_impulse.data(), data_host, _nt);
  TMP::MemSpaceHost::release(data_host);
}

/**
 * @brief Define a layer of constant velocity=v, between the depths [izmin -
 * izmax). This routine allows to create stratified media.
 *
 */
template <class ExecSpace>
void WaveSimulator<ExecSpace>::set_velocity_layer(size_t izmin, size_t izmax,
                                                  float_type v) {
  // assert validity of the zmin and zmax ranges
  assert(izmax <= _nz);
  assert(izmin <= _nz && izmin <= izmax);

  // allocate array on host
  float_type *data_host;
  size_t nelems = (izmax - izmin) * _nx;
  TMP::MemSpaceHost::allocate(&data_host, nelems);

  // set the specified velocity value
  size_t offset = izmin * _nx;
  for (size_t i(0); i < nelems; ++i) data_host[i] = v;

  // calculate offset on array on MemSpace, and then copy the data
  float_type *offset_mem_space_ptr = velmodel.get_ptr() + offset;
  MemSpace::copyFromHost(offset_mem_space_ptr, data_host, nelems);

  // deallocate the host-array
  TMP::MemSpaceHost::release(data_host);

  if (v > _vmax) _vmax = v;

  if (v < _vmin) _vmin = v;
}

/**
 * @brief Store the underlying velocity model in a plain binary file with the
 * provided filename.
 *
 */
template <class ExecSpace>
void WaveSimulator<ExecSpace>::store_velmodel_to_binary(
    const char *filename) const {
  std::fstream file;
  file.open(filename, std::ios_base::out | std::ios_base::binary);

  if (!file.is_open()) throw std::runtime_error("Unable to open file!");

  // allocate array on host, copy the data into, and then store to binary file
  float_type *data_host;
  TMP::MemSpaceHost::allocate(&data_host, velmodel.get_nElems());
  MemSpace::copyToHost(data_host, velmodel.get_ptr(), velmodel.get_nElems());

  // write data to binary file
  file.write((char *)data_host, sizeof(float_type) * velmodel.get_nElems());

  // deallocate the host-array and close file
  TMP::MemSpaceHost::release(data_host);
  file.close();
}

/**
 * @brief Store the current wavefield in a plain binary file with the provided
 * filename.
 *
 */
template <class ExecSpace>
void WaveSimulator<ExecSpace>::store_wavefield_to_binary(
    const char *filename) const {
  std::fstream file;
  file.open(filename, std::ios_base::out | std::ios_base::binary);

  if (!file.is_open()) throw std::runtime_error("Unable to open file!");

  // allocate array on host, copy the data into, and then store to binary file
  float_type *data_host;
  TMP::MemSpaceHost::allocate(&data_host, wavefield.get_nElems());
  MemSpace::copyToHost(data_host, wavefield.get_ptr(), wavefield.get_nElems());

  // write data to binary file
  file.write((char *)data_host, sizeof(float_type) * wavefield.get_nElems());

  // deallocate the host-array and close file
  TMP::MemSpaceHost::release(data_host);
  file.close();
}

/**
 * @brief Return the Courant-Friedricks-Lewy stability condition.
 *
 */
template <class ExecSpace>
float_type WaveSimulator<ExecSpace>::get_CFL_condition() const {
  return _vmax * _dt / _dh;
}

/**
 * @brief Compute and print on screen the Courant-Friedricks-Lewy stability
 * condition.
 *
 */
template <class ExecSpace>
void WaveSimulator<ExecSpace>::print_CFL_condition() const {
  std::cout << "CLF condition: " << get_CFL_condition() << ", ";
  std::cout << "(vmax=" << _vmax << ", dt=" << _dt << ", dh=" << _dh << ")";
  std::cout << std::endl;
}

template <class ExecSpace>
float_type WaveSimulator<ExecSpace>::get_time_step() const {
  return _dt;
}

template <class ExecSpace>
float_type WaveSimulator<ExecSpace>::get_space_step() const {
  return _dh;
}

template <class ExecSpace>
size_t WaveSimulator<ExecSpace>::get_number_of_time_steps() const {
  return _nt;
}

template <class ExecSpace>
size_t WaveSimulator<ExecSpace>::get_source_position_x() const {
  return _srcx;
}

template <class ExecSpace>
size_t WaveSimulator<ExecSpace>::get_source_position_z() const {
  return _srcz;
}

template <class ExecSpace>
size_t WaveSimulator<ExecSpace>::get_dim_nx() const {
  return _nx;
}

template <class ExecSpace>
size_t WaveSimulator<ExecSpace>::get_dim_nz() const {
  return _nz;
}

template <class ExecSpace>
float_type WaveSimulator<ExecSpace>::get_vmin() const {
  return _vmin;
}

template <class ExecSpace>
float_type WaveSimulator<ExecSpace>::get_vmax() const {
  return _vmax;
}