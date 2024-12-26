#ifndef ELASTIC_2D_AXISYMM_HPP
#define ELASTIC_2D_AXISYMM_HPP

#include "ppt/containers/ScalarField.hpp"

enum class source_type {
    ur,
    uz
};

template<typename MemSpace>
struct Model {
    ScalarField<MemSpace> *cp = nullptr;
    ScalarField<MemSpace> *cs = nullptr;
    ScalarField<MemSpace> *rho = nullptr;
    ScalarField<MemSpace> *lam = nullptr;
    ScalarField<MemSpace> *mu = nullptr;
};

template<typename MemSpace>
struct FieldGrad {
    ScalarField<MemSpace> *dr_ur = nullptr;
    ScalarField<MemSpace> *dr_uz = nullptr;
    ScalarField<MemSpace> *dz_ur = nullptr;
    ScalarField<MemSpace> *dz_uz = nullptr;
};

template<typename MemSpace>
struct Strain {
    ScalarField<MemSpace> *epsilon_rr = nullptr;
    ScalarField<MemSpace> *epsilon_tt = nullptr;
    ScalarField<MemSpace> *epsilon_zz = nullptr;
    ScalarField<MemSpace> *epsilon_rz = nullptr;
};

template<typename MemSpace>
struct Stress {
    ScalarField<MemSpace> *sigma_rr = nullptr;
    ScalarField<MemSpace> *sigma_tt = nullptr;
    ScalarField<MemSpace> *sigma_zz = nullptr;
    ScalarField<MemSpace> *sigma_rz = nullptr;
};

template<typename MemSpace>
struct Force {
    ScalarField<MemSpace> *F_r = nullptr;
    ScalarField<MemSpace> *F_z = nullptr;
};

template<typename MemSpace>
struct Field {
    ScalarField<MemSpace> *_ur = nullptr;
    ScalarField<MemSpace> *_ur_old = nullptr;
    ScalarField<MemSpace> *_ur_new = nullptr;
    ScalarField<MemSpace> *_uz = nullptr;
    ScalarField<MemSpace> *_uz_old = nullptr;
    ScalarField<MemSpace> *_uz_new = nullptr;
};

template<class ExecSpace, class MemSpace> int inject_source( Field<MemSpace> const& fields, float_type value, int iz, int ir, ExecSpace tag);

template<class ExecSpace, class MemSpace> int compute_field_gradients( FieldGrad<MemSpace> & dispDerivatives, Field<MemSpace> const& fields,
    float_type dz, float_type dr, ExecSpace tag);

template<class ExecSpace, class MemSpace> int compute_stains( Strain<MemSpace> & strains, FieldGrad<MemSpace> const& dispDerivatives, Field<MemSpace> const& fields, float_type dr, ExecSpace tag);

template<class ExecSpace, class MemSpace> int compute_stresses( Stress<MemSpace> & stresses, Strain<MemSpace> const& strains, FieldGrad<MemSpace> const& dispDerivatives, 
    Field<MemSpace> const& fields, Model<MemSpace> const& models, float_type dr, ExecSpace tag);

template<class ExecSpace, class MemSpace> int compute_forces( Force<MemSpace> & forces, Stress<MemSpace> const& stresses, float_type dz, float_type dr, ExecSpace tag);

template<class ExecSpace, class MemSpace> int compute_time_update( Field<MemSpace> & fields, Force<MemSpace> const& forces, Model<MemSpace> const& models, float_type dt, ExecSpace tag);

#endif