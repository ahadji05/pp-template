
#include "routines/add_source.hpp"
#include "test_defs.hpp"

/**
 * @brief Construct a new TEST object
 *
 * Unit-test for routine:
 *              add_source(p, 5.5, 5, 1, exec_space());
 *
 * p -> ScalarField of dimensionality 3 x 50; initially all values equal to 1.0
 * source amplitude -> 5.5
 * ix -> 5
 * iz -> 1
 *
 * After routine is applied only the value at position = iz * nx + ix should be changed!
 * (in this experiment position = 55)
 *
 * *all other positions: 0,1,...54, ,56,...,149 should remain unchanged.
 */
TEST(test_source_position_value, GTest_add_source)
{
    size_t N = 150;
    TMP::Vector<float_type, memo_space> v(N); // vector of 150 elements
    v.fill(1.0);                              // set each value in vector to 1.0
    ScalarField<memo_space> p(3, 50,
                              v); // construct ScalarField p of dim: nz(3), nx(50) with vector the underlying container

    add_source(p, 5.5, 5, 1, exec_space()); // apply routine

    float_type *host_array;
    TMP::MemSpaceHost::allocate(&host_array, N);               // allocate a host-array for the unit-test assertions
    TMP::MemSpaceHost::copyToHost(host_array, p.get_ptr(), N); // copy data from generic container to host_array
    ASSERT_FLOAT_EQ(host_array[1 * 50 + 5], 5.5);              // source should be added at this particular position
    for (size_t i(0); i < N; ++i)
        if (i != 55)
            ASSERT_FLOAT_EQ(host_array[i], 1.0); // all the rest values should remain unchanged
    TMP::MemSpaceHost::release(host_array);      // release host-array memory
}