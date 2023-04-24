
#include "ppt/routines/time_extrap.hpp"
#include "test_defs.hpp"

/**
 * @brief Test routine fd_time_extrap
 *
 *  Pnew = (2*P - Pold) + (dt/dh)^2 * v(z,x)^2 * (Pzz + Pxx)
 *
 *  P      ->   1.0
 *  Pold   ->   0.5
 *  Pzz    ->   0.5
 *  Pxx    ->   0.5
 *  dt     ->   0.001
 *  dh     ->   2.5
 *  v      ->   1500
 *
 *  Pnew -> (2.0*1.0 - 0.5) + 0.36*(0.5 + 0.5) = 1.5+0.36 = 1.86
 */
TEST(test_fd_time_extrap, GTest_time_extrap)
{
    size_t nz = 3;
    size_t nx = 4;
    ppt::Vector<float_type, memo_space> v(nz * nx);
    v.fill(0.0);
    ScalarField<memo_space> Pnew(nz, nx, v); // Pnew -> 0.0
    v.fill(1.0);
    ScalarField<memo_space> P(nz, nx, v); // P    -> 1.0
    v.fill(0.5);
    ScalarField<memo_space> Pold(nz, nx, v); // Pold -> 0.5
    v.fill(0.5);
    ScalarField<memo_space> Pxx(nz, nx, v); // Pxx -> 0.0
    ScalarField<memo_space> Pzz(nz, nx, v); // Pzz -> 0.0
    v.fill(1500.0);
    ScalarField<memo_space> Vmodel(nz, nx, v); // Vmodel -> 1500 m/s

    float_type dt = 0.001;
    float_type dh = 2.5;

    float_type *host_array;
    ppt::MemSpaceHost::allocate(&host_array, nz * nx);

    fd_time_extrap(Pnew, P, Pold, Pxx, Pzz, Vmodel, dt, dh, exec_space());
    memo_space::copyToHost(host_array, Pnew.get_ptr(), nz * nx);

    for (size_t i(0); i < Pnew.get_nElems(); ++i)
        ASSERT_FLOAT_EQ(host_array[i],
                        1.86); // ASSERT THAT ALL VALUES ARE EQUAL TO 1.86

    ppt::MemSpaceHost::release(host_array);
}