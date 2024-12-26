
#include "ppt/containers/elastic2dAxiSymm.hpp"

// linearized index
#define LID(iz,ir,nr) iz * nr + ir 

// 1st-order derivatives using a 4th-order central-difference stencil
#define d_dR(P,iz,ir,nr,dr) ( ( -1 * ( P[LID(iz,ir+2,nr)] - P[LID(iz,ir-2,nr)] ) + 8 * ( P[LID(iz,ir+1,nr)] - P[LID(iz,ir-1,nr)] ) ) / ( 12 * dr ) )
#define d_dZ(P,iz,ir,nr,dz) ( ( -1 * ( P[LID((iz+2),ir,nr)] - P[LID((iz-2),ir,nr)] ) + 8 * ( P[LID((iz+1),ir,nr)] - P[LID((iz-1),ir,nr)] ) ) / ( 12 * dz ) )


template<>
int inject_source( Field<ppt::MemSpaceHost> const& fields, float_type value, int iz, int ir, ppt::ExecutionSpaceSerial ){
    int nr = fields._ur->get_nx();
    fields._uz->get_ptr()[LID( iz, ir, nr )] = 3*value/4;
    // fields._ur->get_ptr()[LID( iz, ir, nr )] = 1*value/4;
    return 0;
}


template<>
int compute_field_gradients( FieldGrad<ppt::MemSpaceHost> & dispDerivatives, Field<ppt::MemSpaceHost> const& fields,
    float_type dz, float_type dr, ppt::ExecutionSpaceSerial ){

    int nz = dispDerivatives.dr_ur->get_nz();
    int nr = dispDerivatives.dr_ur->get_nx();

    float_type const* const ur = fields._ur->get_ptr();
    float_type const* const uz = fields._uz->get_ptr();
    
    float_type *const dr_ur = dispDerivatives.dr_ur->get_ptr();
    float_type *const dr_uz = dispDerivatives.dr_uz->get_ptr();
    float_type *const dz_ur = dispDerivatives.dz_ur->get_ptr();
    float_type *const dz_uz = dispDerivatives.dz_uz->get_ptr();

    for (int iz(2); iz < nz - 2; ++iz)
        for (int ir(2); ir < nr - 2; ++ir){
            dr_ur[LID( iz, ir, nr )] = d_dR( ur, iz, ir, nr, dr );
            dr_uz[LID( iz, ir, nr )] = d_dR( uz, iz, ir, nr, dr );
            dz_ur[LID( iz, ir, nr )] = d_dZ( ur, iz, ir, nr, dz );
            dz_uz[LID( iz, ir, nr )] = d_dZ( uz, iz, ir, nr, dz );
        }

    return 0;
}


template<>
int compute_stains( Strain<ppt::MemSpaceHost> & strains, FieldGrad<ppt::MemSpaceHost> const& dispDerivatives, Field<ppt::MemSpaceHost> const& fields, 
    float_type dr, ppt::ExecutionSpaceSerial ){

    int nz = strains.epsilon_tt->get_nz();
    int nr = strains.epsilon_tt->get_nx();

    float_type const* const ur    = fields._ur->get_ptr();
    float_type const* const dz_ur = dispDerivatives.dz_ur->get_ptr();
    float_type const* const dr_uz = dispDerivatives.dr_uz->get_ptr();

    float_type *const epsilon_tt = strains.epsilon_tt->get_ptr();
    float_type *const epsilon_rz = strains.epsilon_rz->get_ptr();

    for (int iz(2); iz < nz - 2; ++iz)
        for (int ir(2); ir < nr - 2; ++ir){
            float_type r = ir * dr + 1e-5;
            epsilon_tt[LID( iz, ir, nr )] = ur[LID( iz, ir, nr )] / r;
            epsilon_rz[LID( iz, ir, nr )] = 0.5 * ( dz_ur[LID( iz, ir, nr )] + dr_uz[LID( iz, ir, nr )] );
        }

    return 0;
}


template<>
int compute_stresses( Stress<ppt::MemSpaceHost> & stresses, Strain<ppt::MemSpaceHost> const& strains, FieldGrad<ppt::MemSpaceHost> const& dispDerivatives, 
    Field<ppt::MemSpaceHost> const& fields, Model<ppt::MemSpaceHost> const& models, float_type dr, ppt::ExecutionSpaceSerial ){

    int nz = stresses.sigma_rr->get_nz();
    int nr = stresses.sigma_rr->get_nx();

    float_type const* const ur         = fields._ur->get_ptr();
    float_type const* const dr_ur      = dispDerivatives.dr_ur->get_ptr();
    float_type const* const dz_uz      = dispDerivatives.dz_uz->get_ptr();
    float_type const* const epsilon_rr = strains.epsilon_rr->get_ptr();
    float_type const* const epsilon_zz = strains.epsilon_zz->get_ptr();
    float_type const* const epsilon_tt = strains.epsilon_tt->get_ptr();
    float_type const* const epsilon_rz = strains.epsilon_rz->get_ptr();
    float_type const* const lam        = models.lam->get_ptr();
    float_type const* const mu         = models.mu->get_ptr();

    float_type *const sigma_rr = stresses.sigma_rr->get_ptr();
    float_type *const sigma_zz = stresses.sigma_zz->get_ptr();
    float_type *const sigma_tt = stresses.sigma_tt->get_ptr();
    float_type *const sigma_rz = stresses.sigma_rz->get_ptr();

    for (int iz(2); iz < nz - 2; ++iz)
        for (int ir(2); ir < nr - 2; ++ir){
            float_type r = ir * dr + 1e-5;
            float_type common_term = lam[LID( iz, ir, nr )] * ( dr_ur[LID( iz, ir, nr )] + ( ur[LID( iz, ir, nr )] / r ) + dz_uz[LID( iz, ir, nr )] );
            sigma_rr[LID( iz, ir, nr )] = common_term + 2 * mu[LID( iz, ir, nr )] * epsilon_rr[LID( iz, ir, nr )];
            sigma_zz[LID( iz, ir, nr )] = common_term + 2 * mu[LID( iz, ir, nr )] * epsilon_zz[LID( iz, ir, nr )];
            sigma_tt[LID( iz, ir, nr )] = common_term + 2 * mu[LID( iz, ir, nr )] * epsilon_tt[LID( iz, ir, nr )];
            sigma_rz[LID( iz, ir, nr )] = 2 * mu[LID( iz, ir, nr )] * epsilon_rz[LID( iz, ir, nr )];
        }

    return 0;
}


template<>
int compute_forces( Force<ppt::MemSpaceHost> & forces, Stress<ppt::MemSpaceHost> const& stresses, float_type dz, float_type dr, ppt::ExecutionSpaceSerial ){

    int nz = forces.F_z->get_nz();
    int nr = forces.F_r->get_nx();

    float_type const* const sigma_rr = stresses.sigma_rr->get_ptr();
    float_type const* const sigma_zz = stresses.sigma_zz->get_ptr();
    float_type const* const sigma_tt = stresses.sigma_tt->get_ptr();
    float_type const* const sigma_rz = stresses.sigma_rz->get_ptr();

    float_type *const Fr = forces.F_r->get_ptr();
    float_type *const Fz = forces.F_z->get_ptr();

    for (int iz(2); iz < nz - 2; ++iz)
        for (int ir(2); ir < nr - 2; ++ir){
            float_type r = ir * dr + 1e-5;
            Fr[LID( iz, ir, nr )] = d_dR( sigma_rr, iz, ir, nr, dr ) + ( ( sigma_rr[LID( iz, ir, nr )] - sigma_tt[LID( iz, ir, nr )] ) / r ) + d_dZ( sigma_rz, iz, ir, nr, dz );
            Fz[LID( iz, ir, nr )] = d_dZ( sigma_zz, iz, ir, nr, dz ) + ( sigma_rz[LID( iz, ir, nr )] / r ) + d_dR( sigma_rz, iz, ir, nr, dr );
        }

    return 0;
}


template<>
int compute_time_update( Field<ppt::MemSpaceHost> & fields, Force<ppt::MemSpaceHost> const& forces, Model<ppt::MemSpaceHost> const& models, float_type dt, ppt::ExecutionSpaceSerial ){

    int nz = fields._ur_new->get_nz();
    int nr = fields._ur_new->get_nx();

    float_type const* const ur     = fields._ur->get_ptr();
    float_type const* const uz     = fields._uz->get_ptr();
    float_type const* const ur_old = fields._ur_old->get_ptr();
    float_type const* const uz_old = fields._uz_old->get_ptr();
    float_type const* const rho    = models.rho->get_ptr();
    float_type const*const Fr      = forces.F_r->get_ptr();
    float_type const*const Fz      = forces.F_z->get_ptr();

    float_type *const uz_new = fields._uz_new->get_ptr();
    float_type *const ur_new = fields._ur_new->get_ptr();

    for (int iz(2); iz < nz - 2; ++iz)
        for (int ir(2); ir < nr - 2; ++ir){
            ur_new[LID( iz, ir, nr )] = 2 * ur[LID( iz, ir, nr )] - ur_old[LID( iz, ir, nr )] + ( dt * dt * Fr[LID( iz, ir, nr )]  / rho[LID( iz, ir, nr )] );
            uz_new[LID( iz, ir, nr )] = 2 * uz[LID( iz, ir, nr )] - uz_old[LID( iz, ir, nr )] + ( dt * dt * Fz[LID( iz, ir, nr )]  / rho[LID( iz, ir, nr )] );
        }

    return 0;
}
