
#include "algorithms/WaveSimulator.hpp"
#include <iostream>

int main()
{
    // Define the WaveSimulator based on the exec_space from "types.hpp"
    WaveSimulator<exec_space> Sim;

    // Set modelling parameters
    Sim.set_time_step(0.0005);
    Sim.set_number_of_time_steps(2001);
    Sim.set_dimensions(701, 2001);
    Sim.set_space_step(7.5);
    Sim.set_source_position_z(250);
    Sim.set_source_position_x(1000);
    Sim.make_ricker(10);

    // Set the background velocity and add layers of different velocities
    Sim.set_vmin(1500);
    Sim.set_velocity_layer(300, 450, 3000);
    Sim.set_velocity_layer(450, 701, 4800);

    // Compute and print the Courant-Friedricks-Lewy condition:
    Sim.print_CFL_condition();

    // run simulation for all time-steps
    Sim.run();

    // Output velocity model and final wavefield in plain binary files
    Sim.store_velmodel_to_binary("velocity_model.bin");
    Sim.store_wavefield_to_binary("wavefield.bin");

    return 0;
}