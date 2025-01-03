
#include "ppt/algorithms/WaveSimulator.hpp"
#include <iostream>

// compute-sanitizer --tool memcheck --leak-check=full ./main_exe
// valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./main_exe

int main()
{
    // Define the WaveSimulator based on the exec_space from "types.hpp"
    WaveSimulator<exec_space> Sim;

    // Set modelling parameters
    Sim.set_time_step(0.0002);
    Sim.set_number_of_time_steps(4001);
    Sim.set_dimensions(1751, 5001);
    Sim.set_space_step(3.00);
    Sim.set_source_position_z(625);
    Sim.set_source_position_x(2500);
    Sim.make_ricker(10);

    // Set the background velocity and add layers of different velocities
    Sim.set_vmin(1500);
    Sim.set_velocity_layer(750, 1125, 3000);
    Sim.set_velocity_layer(1125, 1751, 4800);

    // Compute and print the Courant-Friedricks-Lewy condition:
    Sim.print_CFL_condition();

    // run simulation for all time-steps
    Sim.run();

    // Output velocity model and final wavefield in plain binary files
    Sim.store_velmodel_to_binary("velocity_model.bin");
    Sim.store_wavefield_to_binary("wavefield.bin");

    return 0;
}