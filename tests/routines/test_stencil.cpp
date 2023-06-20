
#include "ppt/routines/stencil.hpp"
#include "test_defs.hpp"

/**
 *        INPUT P                   OUTPUT Pxx
 *        -------                   ----------
 *
 *      0 0 1 0 0 0          0   0   0   0   0   0
 *      0 0 0 0 0 0          0   0   0   0   0   0
 * p -> 0 0 1 0 0 1   pxx->  0   0 -2.5 1.25 0   0
 *      1 0 0 1 0 0          0   0   0   0   0   0
 *      0 0 0 0 0 0          0   0   0   0   0   0
 */
TEST(test_fd_pxx, GTest_stencil)
{
    size_t nz = 5;
    size_t nx = 6;
    ppt::Vector<float_type, memo_space> v(nz * nx);
    v.fill(0);
    ScalarField<memo_space> p(nz, nx, v);
    ScalarField<memo_space> pxx(p);

    float_type *host_array;
    ppt::MemSpaceHost::allocate(&host_array, nz * nx);
    for (size_t i(0); i < nx * nz; ++i)
        host_array[i] = 0;
    host_array[2] = 1;
    host_array[14] = 1;
    host_array[17] = 1;
    host_array[18] = 1;
    host_array[21] = 1;

    memo_space::copyFromHost(p.get_ptr(), host_array, nz * nx);
    fd_pxx(pxx, p, exec_space());
    memo_space::copyToHost(host_array, pxx.get_ptr(), nz * nx);
    
    for(size_t i(0); i < nx * nz; ++i)
    {
      if(i == 14) { // == -5/2 = -2.5
        ASSERT_FLOAT_EQ(host_array[14], float_type(-5.0 / (float_type)2.0));
      } 
        else if (i == 15) // == 4/3 - 1/12 = 1.25
      {
        ASSERT_FLOAT_EQ(host_array[15],  (float_type)(4.0/3.0) + (float_type)(-1.0/12.0));
      } 
        else // == 0
      {
        ASSERT_FLOAT_EQ(host_array[i], 0);
      }
    }

    ppt::MemSpaceHost::release(host_array);
}



/**
 *        INPUT P                   OUTPUT Pzz
 *        -------                   ----------
 *
 *      0 0 1 0 0 0          0   0   0      0   0   0
 *      0 0 0 0 0 0          0   0   0      0   0   0
 * p -> 0 0 1 0 0 1   pzz->  0   0 -2.5833 4/3  0   0
 *      1 0 0 1 0 0          0   0   0      0   0   0
 *      0 0 0 0 0 0          0   0   0      0   0   0
 */
TEST(test_fd_pzz, GTest_stencil)
{
    size_t nz = 5;
    size_t nx = 6;
    ppt::Vector<float_type, memo_space> v(nz * nx);
    v.fill(0);
    ScalarField<memo_space> p(nz, nx, v);
    ScalarField<memo_space> pzz(p);

    float_type *host_array;
    ppt::MemSpaceHost::allocate(&host_array, nz * nx);
    for (size_t i(0); i < nx * nz; ++i)
        host_array[i] = 0;
    host_array[2] = 1;
    host_array[14] = 1;
    host_array[17] = 1;
    host_array[18] = 1;
    host_array[21] = 1;


    memo_space::copyFromHost(p.get_ptr(), host_array, nz * nx);
    fd_pzz(pzz, p, exec_space());
    memo_space::copyToHost(host_array, pzz.get_ptr(), nz * nx);

    for(size_t i(0); i < nx * nz; ++i)
    {
      if(i == 14) { // == -5/2 -1/12 = -2.5833333
        ASSERT_FLOAT_EQ(host_array[14], (float_type)(-5.0/2.0) + (float_type)(-1.0/12.0));
      }
        else if (i == 15) // == 4/3 = 1.333333
      {
        ASSERT_FLOAT_EQ(host_array[15],  (float_type)(4.0/3.0) );
      } 
        else // == 0
      {
        ASSERT_FLOAT_EQ(host_array[i], 0);
      }
    }
    
    ppt::MemSpaceHost::release(host_array);
}
