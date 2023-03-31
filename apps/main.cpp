
#include <iostream>

#include "algorithms/WaveSimulator.hpp"

int main()
{
    // Define the WaveSimulator
    WaveSimulator<TMP::ExecutionSpaceSerial> Sim;

    // Set modelling parameters
    Sim.set_time_step(0.001);
    Sim.set_number_of_time_steps(1001);
    Sim.set_dimensions(701, 2001);
    Sim.set_space_step(7.5);
    Sim.set_source_position_z(250);
    Sim.set_source_position_x(1000);
    Sim.make_ricker(10);

    // Set the background velocity and add layers of different velocities
    Sim.set_vmin(1500);
    Sim.set_velocity_layer(300, 450, 3000);
    Sim.set_velocity_layer(450, 701, 4800);

    // Compute the Courant-Friedrichs-Lewy condition:
    std::cout << "CLF condition: " << Sim.CLF_condition() << std::endl;

    // run simulation for all time-steps
    Sim.run();

    // Output velocity model and final wavefield in plain binary files
    Sim.store_velmodel_to_binary("velocity_model.bin");
    Sim.store_wavefield_to_binary("wavefield.bin");

    return 0;
}