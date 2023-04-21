
#include "ppt/algorithms/WaveSimulator.hpp"
#include "test_defs.hpp"

TEST(test_public_methods, GTest_WaveSimulator) {
  WaveSimulator<exec_space> Sim;

  Sim.set_source_position_x(16);
  ASSERT_EQ(Sim.get_source_position_x(), 16);

  Sim.set_source_position_z(1);
  ASSERT_EQ(Sim.get_source_position_z(), 1);

  Sim.set_time_step(0.16);
  ASSERT_FLOAT_EQ(Sim.get_time_step(), 0.16);

  Sim.set_space_step(1.6);
  ASSERT_FLOAT_EQ(Sim.get_space_step(), 1.6);

  Sim.set_dimensions(10, 5);
  ASSERT_EQ(Sim.get_dim_nz(), 10);
  ASSERT_EQ(Sim.get_dim_nx(), 5);

  Sim.set_vmin(1500);
  Sim.set_velocity_layer(3, 7, 4800);

  float_type clf = 4800.0 * 0.16 / 1.6;
  ASSERT_FLOAT_EQ(Sim.get_CFL_condition(), clf);
}