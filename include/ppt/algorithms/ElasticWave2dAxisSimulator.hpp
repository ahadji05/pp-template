#ifndef PPT_ELASTIC_WAVE_2D_AXIS_SIMULATOR_HPP
#define PPT_ELASTIC_WAVE_2D_AXIS_SIMULATOR_HPP

#include "ppt/containers/elastic2dAxiSymm.hpp"
#include <fstream>
#include <vector>

/**
 * @brief This class implements ...
 */
template <class ExecSpace> 
class ElasticWave2dAxiSymmetricSimulator {
  private:
    using MemSpace = typename ExecSpace::accessible_space;

    FieldGrad<MemSpace> _fieldGrads;
    Field<MemSpace>  _fields;
    Stress<MemSpace> _stresses;
    Strain<MemSpace> _strains;
    Force<MemSpace>  _forces;

    size_t _nt;
    size_t _nz;
    size_t _nr;
    float_type _dt;
    float_type _dz;
    float_type _dr;
    float_type _srcz;
    float_type _srcr;
    std::vector<float_type> _wavelet;

  public:
    ElasticWave2dAxiSymmetricSimulator() = default;
    ~ElasticWave2dAxiSymmetricSimulator() = default;

    // Set-methods
    void set_time_step(float_type dt)   { _dt = dt; }
    void set_radial_step(float_type dr) { _dr = dr; }
    void set_depth_step(float_type dz)  { _dz = dz; }
    void set_number_of_time_steps(size_t nt)  { _nt = nt; }
    void set_dimensions(size_t nz, size_t nr) { _nz = nz; _nr = nr; }
    void set_source_position_r(float_type r)  { _srcr = r; }
    void set_source_position_z(float_type z)  { _srcz = z; }

    // Get-methods
    float_type get_time_step() const   { return _dt; }
    float_type get_radial_step() const { return _dr; }
    float_type get_depth_step() const  { return _dz; }
    size_t get_number_of_time_steps() const   { return _nt; }
    size_t get_number_of_radial_steps() const { return _nr; }
    size_t get_number_of_depth_steps() const  { return _nz; }
    size_t get_source_position_r() const { return _srcr; }
    size_t get_source_position_z() const { return _srcz; }
    float_type get_CFL_condition() const { return 0; }

    Field<MemSpace>     const* get_field()     const { return &_fields;     }
    FieldGrad<MemSpace> const* get_fieldGrad() const { return &_fieldGrads; }
    Strain<MemSpace>    const* get_strain()    const { return &_strains;    }
    Stress<MemSpace>    const* get_stress()    const { return &_stresses;   }
    Force<MemSpace>     const* get_force()     const { return &_forces;     }

    // Other public methods
    void make_ricker(float_type fpeak);
    bool wrong_initialization_of_simulator_encountered() { return false; }
    void allocate_internal_data_structures();
    void clean_internal_data_structures();

    // main algorithm
    int run( Model<MemSpace> const& models ){
        if ( wrong_initialization_of_simulator_encountered() ) return 1;
        for (size_t i(0); i < _nt; ++i) {
            if (i % 250 == 0) std::cout << "time-step: " << i << std::endl;
            if ( inject_source( _fields, _wavelet[i], (int)(_srcz/_dz), (int)(_srcr/_dr), ExecSpace() ) )   return 1;
            if ( compute_field_gradients( _fieldGrads, _fields, _dz, _dr, ExecSpace() ) )                   return 1;
            if ( compute_stains( _strains, _fieldGrads, _fields, _dr, ExecSpace() ) )                       return 1;
            if ( compute_stresses( _stresses, _strains, _fieldGrads, _fields, models, _dr, ExecSpace() ) )  return 1;
            if ( compute_forces( _forces, _stresses, _dz, _dr, ExecSpace() ) )                              return 1;
            if ( compute_time_update( _fields, _forces, models, _dt, ExecSpace() ) )                        return 1;
            std::swap( _fields._ur_old , _fields._ur );
            std::swap( _fields._ur , _fields._ur_new );
            std::swap( _fields._uz_old , _fields._uz );
            std::swap( _fields._uz , _fields._uz_new );
        }
        return 0;
    }
};

/**
 * @brief Create a Ricker-wavelet:
 * R(t) = (1 - 2 pi^2 fpeak^2 t^2) * exp( -1 * pi^2 fpeak^2 t^2 )
 */
template <class ExecSpace> void ElasticWave2dAxiSymmetricSimulator<ExecSpace>::make_ricker(float_type fpeak)
{
    _wavelet.resize(_nt);
    // compute Ricker-wavelet with user defined peak frequency
    _wavelet[0] = 1.0;
    for (int it = 1; it < _nt; ++it)
    {
        float_type t         = it * _dt;
        float_type term      = M_PI * M_PI * fpeak * fpeak * t * t;
        _wavelet[it]         = (1 - 2 * term) * exp(-term);
    }
}

template <class ExecSpace> void ElasticWave2dAxiSymmetricSimulator<ExecSpace>::allocate_internal_data_structures()
{
    // Fields
    _fields._ur     = new ScalarField<MemSpace>( _nz, _nr );
    _fields._uz     = new ScalarField<MemSpace>( _nz, _nr );
    _fields._ur_new = new ScalarField<MemSpace>( _nz, _nr );
    _fields._ur_old = new ScalarField<MemSpace>( _nz, _nr );
    _fields._uz_new = new ScalarField<MemSpace>( _nz, _nr );
    _fields._uz_old = new ScalarField<MemSpace>( _nz, _nr );

    // Field gradients
    _fieldGrads.dr_ur = new ScalarField<MemSpace>( _nz, _nr );
    _fieldGrads.dr_uz = new ScalarField<MemSpace>( _nz, _nr );
    _fieldGrads.dz_ur = new ScalarField<MemSpace>( _nz, _nr );
    _fieldGrads.dz_uz = new ScalarField<MemSpace>( _nz, _nr );

    // Strains
    _strains.epsilon_rr = _fieldGrads.dr_ur;
    _strains.epsilon_zz = _fieldGrads.dz_uz;
    _strains.epsilon_tt = new ScalarField<MemSpace>( _nz, _nr );
    _strains.epsilon_rz = new ScalarField<MemSpace>( _nz, _nr );

    // Stresses
    _stresses.sigma_rr = new ScalarField<MemSpace>( _nz, _nr );
    _stresses.sigma_zz = new ScalarField<MemSpace>( _nz, _nr );
    _stresses.sigma_tt = new ScalarField<MemSpace>( _nz, _nr );
    _stresses.sigma_rz = new ScalarField<MemSpace>( _nz, _nr );

    // Forces
    _forces.F_r = new ScalarField<MemSpace>( _nz, _nr );
    _forces.F_z = new ScalarField<MemSpace>( _nz, _nr );
}

template <class ExecSpace> void ElasticWave2dAxiSymmetricSimulator<ExecSpace>::clean_internal_data_structures()
{
    // Fields
    delete _fields._ur; _fields._ur = nullptr;
    delete _fields._uz; _fields._uz = nullptr;
    delete _fields._ur_new; _fields._ur_new = nullptr;
    delete _fields._ur_old; _fields._ur_old = nullptr;
    delete _fields._uz_new; _fields._uz_new = nullptr;
    delete _fields._uz_old; _fields._uz_old = nullptr;

    // Field gradients
    delete _fieldGrads.dr_ur; _fieldGrads.dr_ur = nullptr;
    delete _fieldGrads.dr_uz; _fieldGrads.dr_uz = nullptr;
    delete _fieldGrads.dz_ur; _fieldGrads.dz_ur = nullptr;
    delete _fieldGrads.dz_uz; _fieldGrads.dz_uz = nullptr;

    // Strains
    delete _strains.epsilon_tt; _strains.epsilon_tt = nullptr;
    delete _strains.epsilon_rz; _strains.epsilon_rz = nullptr;
    // _strains.epsilon_rr points to _fieldGrads.dr_ur (it has already been deleted)
    // _strains.epsilon_zz points to _fieldGrads.dz_uz (it has already been deleted)

    // Stresses
    delete _stresses.sigma_rr; _stresses.sigma_rr = nullptr;
    delete _stresses.sigma_zz; _stresses.sigma_zz = nullptr;
    delete _stresses.sigma_tt; _stresses.sigma_tt = nullptr;
    delete _stresses.sigma_rz; _stresses.sigma_rz = nullptr;

    // Forces
    delete _forces.F_r; _forces.F_r = nullptr;
    delete _forces.F_z; _forces.F_z = nullptr;
}


template<typename MemSpace>
void store_to_binary(const char *filename, ScalarField<MemSpace> const* const field) {
    std::fstream file;
    file.open(filename, std::ios_base::out | std::ios_base::binary);

    if (!file.is_open()) throw std::runtime_error("Unable to open file!");

    // allocate array on host, copy the data into, and then store to binary file
    float_type *data_host;
    ppt::MemSpaceHost::allocate(&data_host, field->get_nElems());
    MemSpace::copyToHost(data_host, field->get_ptr(), field->get_nElems());

    // write data to binary file
    file.write((char *)data_host, sizeof(float_type) * field->get_nElems());

    // deallocate the host-array and close file
    ppt::MemSpaceHost::release(data_host);
    file.close();
}

#endif // PPT_WAVE_SIMULATOR_HPP