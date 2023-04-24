
#include "ppt/routines/stencil.hpp"
#include "test_defs.hpp"

/**
 *    INPUT P              OUTPUT Pxx
 *    -------              ----------
 *
 *      0   1                0    0
 *      0   0                0    0
 * p -> 0   0      pxx ->    4/3 -1/12
 *      1   0                -5/2 0
 *      0   0                0    0
 *      0   0                0    0
 *
 */
TEST(test_fd_pxx, GTest_stencil)
{
    size_t nz = 2;
    size_t nx = 6;
    ppt::Vector<float_type, memo_space> v(nz * nx);
    v.fill(0);
    ScalarField<memo_space> p(nz, nx, v);
    ScalarField<memo_space> pxx(p);

    float_type *host_array;
    ppt::MemSpaceHost::allocate(&host_array, nz * nx);
    for (size_t i(0); i < nx * nz; ++i)
        host_array[i] = 0;
    host_array[3] = 1;
    host_array[6] = 1;

    memo_space::copyFromHost(p.get_ptr(), host_array, nz * nx);
    fd_pxx(pxx, p, exec_space());
    memo_space::copyToHost(host_array, pxx.get_ptr(), nz * nx);

    ASSERT_FLOAT_EQ(host_array[2], float_type(4.0 / (float_type)3.0));
    ASSERT_FLOAT_EQ(host_array[3], -float_type(5.0 / (float_type)2.0));
    ASSERT_FLOAT_EQ(host_array[8], -float_type(1.0 / (float_type)12.0));

    ppt::MemSpaceHost::release(host_array);
}

/**
 *    INPUT P              OUTPUT Pzz
 *    -------              ----------
 *
 *      0   0                0    0
 *      0   1                0    0
 * p -> 0   0     pzz ->    4/3  4/3
 *      1   0               -5/2 -1/12
 *      0   0                0    0
 *      0   0                0    0
 *
 */
TEST(test_fd_pzz, GTest_stencil)
{
    size_t nz = 6;
    size_t nx = 2;
    ppt::Vector<float_type, memo_space> v(nz * nx);
    v.fill(0);
    ScalarField<memo_space> p(nz, nx, v);
    ScalarField<memo_space> pzz(p);

    float_type *host_array;
    ppt::MemSpaceHost::allocate(&host_array, nz * nx);
    for (size_t i(0); i < nx * nz; ++i)
        host_array[i] = 0;
    host_array[3] = 1;
    host_array[6] = 1;

    memo_space::copyFromHost(p.get_ptr(), host_array, nz * nx);
    fd_pzz(pzz, p, exec_space());
    memo_space::copyToHost(host_array, pzz.get_ptr(), nz * nx);

    ASSERT_FLOAT_EQ(host_array[4], float_type(4.0 / (float_type)3.0));
    ASSERT_FLOAT_EQ(host_array[5], float_type(4.0 / (float_type)3.0));
    ASSERT_FLOAT_EQ(host_array[6], -float_type(5.0 / (float_type)2.0));
    ASSERT_FLOAT_EQ(host_array[7], -float_type(1.0 / (float_type)12.0));

    ppt::MemSpaceHost::release(host_array);
}