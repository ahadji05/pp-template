

#include "ppt/algorithms/ElasticWave2dAxisSimulator.hpp"
#include <iostream>
#include <cmath>

// compute-sanitizer --tool memcheck --leak-check=full ./main_exe
// valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./main_exe

int main(){
    
    int nz = 400;
    int nr = 1000;
    int nt = 1501;

    // Define model with three layers
    Model<ppt::MemSpaceHost> model;
    model.cp = new ScalarField<ppt::MemSpaceHost>( nz, nr );
    model.cs = new ScalarField<ppt::MemSpaceHost>( nz, nr );
    model.rho = new ScalarField<ppt::MemSpaceHost>( nz, nr );
    model.lam = new ScalarField<ppt::MemSpaceHost>( nz, nr );
    model.mu = new ScalarField<ppt::MemSpaceHost>( nz, nr );
    for(int iz=0; iz < nz/3; ++iz)
        for(int ir=0; ir < nr; ++ir){
            int idx = iz*nr+ir;
            model.cp->get_ptr()[idx]  = 1500;
            model.cs->get_ptr()[idx]  = 0;
            model.rho->get_ptr()[idx] = 1000;
        }

    for(int iz=nz/3-1; iz < nz/2; ++iz)
        for(int ir=0; ir < nr; ++ir){
            int idx = iz*nr+ir;
            model.cp->get_ptr()[idx]  = 3500;
            model.cs->get_ptr()[idx]  = 500;
            model.rho->get_ptr()[idx] = 2500;
        }

    for(int iz=nz/2-1; iz < nz; ++iz)
        for(int ir=0; ir < nr; ++ir){
            int idx = iz*nr+ir;
            model.cp->get_ptr()[idx]  = 4000;
            model.cs->get_ptr()[idx]  = 2000;
            model.rho->get_ptr()[idx] = 1500;
        }

    for(int iz=0; iz < nz; ++iz)
        for(int ir=0; ir < nr; ++ir){
            int idx = iz*nr+ir;
            model.mu->get_ptr()[idx]  = model.rho->get_ptr()[idx] * std::pow(model.cs->get_ptr()[idx],2) / 1;
            model.lam->get_ptr()[idx] = (std::pow(model.cp->get_ptr()[idx],2) / 1) * model.rho->get_ptr()[idx] - 2 * model.mu->get_ptr()[idx];
        }

    store_to_binary( "cp.bin",  model.cp );
    store_to_binary( "cs.bin",  model.cs );
    store_to_binary( "rho.bin", model.rho );
    store_to_binary( "mu.bin",  model.mu );
    store_to_binary( "lam.bin", model.lam );

    // Define the WaveSimulator based on the exec_space from "types.hpp"
    ElasticWave2dAxiSymmetricSimulator<ppt::ExecutionSpaceSerial> Sim;
    Sim.set_dimensions( nz, nr );
    Sim.set_number_of_time_steps( nt );
    Sim.set_radial_step(5);
    Sim.set_depth_step(5);
    Sim.set_time_step(0.0003);
    Sim.make_ricker(10);
    Sim.set_source_position_r(2500);
    Sim.set_source_position_z(15);

    Sim.allocate_internal_data_structures();

    if ( Sim.run( model ) ) return EXIT_FAILURE;

    store_to_binary( "field_ur.bin", Sim.get_field()->_ur );
    store_to_binary( "field_ur_new.bin", Sim.get_field()->_ur_new );
    store_to_binary( "field_ur_old.bin", Sim.get_field()->_ur_old );
    store_to_binary( "field_uz.bin", Sim.get_field()->_uz );
    store_to_binary( "field_uz_new.bin", Sim.get_field()->_uz_new );
    store_to_binary( "field_uz_old.bin", Sim.get_field()->_uz_old );

    store_to_binary( "field_dr_ur.bin", Sim.get_fieldGrad()->dr_ur );
    store_to_binary( "field_dr_uz.bin", Sim.get_fieldGrad()->dr_uz );
    store_to_binary( "field_dz_ur.bin", Sim.get_fieldGrad()->dz_ur );
    store_to_binary( "field_dz_uz.bin", Sim.get_fieldGrad()->dz_uz );

    store_to_binary( "field_eps_rr.bin", Sim.get_strain()->epsilon_rr );
    store_to_binary( "field_eps_zz.bin", Sim.get_strain()->epsilon_zz );
    store_to_binary( "field_eps_tt.bin", Sim.get_strain()->epsilon_tt );
    store_to_binary( "field_eps_rz.bin", Sim.get_strain()->epsilon_rz );

    store_to_binary( "field_sigma_rr.bin", Sim.get_stress()->sigma_rr );
    store_to_binary( "field_sigma_zz.bin", Sim.get_stress()->sigma_zz );
    store_to_binary( "field_sigma_tt.bin", Sim.get_stress()->sigma_tt );
    store_to_binary( "field_sigma_rz.bin", Sim.get_stress()->sigma_rz );

    store_to_binary( "field_Fr.bin", Sim.get_force()->F_r );
    store_to_binary( "field_Fz.bin", Sim.get_force()->F_z );

    ScalarField<ppt::MemSpaceHost> Pwave( nz, nr );
    ScalarField<ppt::MemSpaceHost> Swave( nz, nr );
    for(int iz=0; iz < nz; ++iz)
        for(int ir=1; ir < nr; ++ir){
            Pwave.get_ptr()[iz*nr+ir] = 
                Sim.get_fieldGrad()->dr_ur->get_ptr()[iz*nr+ir] + 
                (Sim.get_field()->_ur->get_ptr()[iz*nr+ir] / ir*Sim.get_radial_step()) +
                Sim.get_fieldGrad()->dz_uz->get_ptr()[iz*nr+ir];

            Swave.get_ptr()[iz*nr+ir] = 
                Sim.get_fieldGrad()->dz_ur->get_ptr()[iz*nr+ir] -
                Sim.get_fieldGrad()->dr_uz->get_ptr()[iz*nr+ir];
        }

    store_to_binary( "Pwave.bin", &Pwave );
    store_to_binary( "Swave.bin", &Swave );

    Sim.clean_internal_data_structures();

    delete model.cp;
    delete model.cs;
    delete model.rho;
    delete model.lam;
    delete model.mu;

    return EXIT_SUCCESS;
}