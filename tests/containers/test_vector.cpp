#include "containers/Vector.hpp"
#include "test_defs.hpp"

TEST(test_constructor_explicit_host_container, GTest_Vector)
{
    TMP::Vector<float, TMP::MemSpaceHost> vA(20);
    ASSERT_EQ(vA.get_nElems(), 20);

    vA.fill(10.5);
    for (size_t i(0); i < vA.get_nElems(); ++i)
        ASSERT_FLOAT_EQ(vA[i], 10.5);
}

TEST(test_constructor, GTest_Vector)
{
    TMP::Vector<float, memo_space> vA(10);
    ASSERT_EQ(vA.get_nElems(), 10);

    vA.resize(12);
    ASSERT_EQ(vA.get_nElems(), 12);
    vA.fill(-0.15);

    float *host_data;
    TMP::MemSpaceHost::allocate(&host_data, 12);
    memo_space::copyToHost(host_data, vA.get_ptr(), 12);

    for (size_t i(0); i < vA.get_nElems(); ++i)
        ASSERT_FLOAT_EQ(host_data[i], -0.15);

    TMP::MemSpaceHost::release(host_data);
}

TEST(test_constructor_two_explicit_host_container, GTest_Vector)
{
    double host_data[9] = {0.01, -0.1, 0, 1.0, 1.1, 2.5, -12.1, 0, 2};
    TMP::Vector<double, TMP::MemSpaceHost> vA(host_data, 9);
    ASSERT_EQ(vA.get_nElems(), 9);
    for (size_t i(0); i < vA.get_nElems(); ++i)
        ASSERT_FLOAT_EQ(vA[i], host_data[i]);
}

TEST(test_constructor_two, GTest_Vector)
{
    double host_data[9] = {0.01, -0.1, 0, 1.0, 1.1, 2.5, -12.1, 0, 2};
    TMP::Vector<double, memo_space> vA(host_data, 9);
    ASSERT_EQ(vA.get_nElems(), 9);

    double *host_data2;
    TMP::MemSpaceHost::allocate(&host_data2, 9);
    memo_space::copyToHost(host_data2, vA.get_ptr(), 9);
    for (size_t i(0); i < vA.get_nElems(); ++i)
        ASSERT_FLOAT_EQ(host_data2[i], host_data[i]);

    TMP::MemSpaceHost::release(host_data2);
}

TEST(test_copy_constructor, GTest_Vector)
{
    TMP::Vector<float, memo_space> vA(1014);
    vA.fill(1.0015);

    TMP::Vector<float, memo_space> vB(vA);
    ASSERT_EQ(vB.get_nElems(), 1014);

    float *host_data;
    TMP::MemSpaceHost::allocate(&host_data, 1014);
    memo_space::copyToHost(host_data, vA.get_ptr(), 1014);
    for (size_t i(0); i < 1014; ++i)
        ASSERT_FLOAT_EQ(host_data[i], 1.0015);

    TMP::MemSpaceHost::release(host_data);
}

TEST(test_copy_assign, GTest_Vector)
{
    TMP::Vector<float, memo_space> vA(124003);
    vA.fill(-1.0015);

    TMP::Vector<float, memo_space> vB(vA);
    ASSERT_EQ(vB.get_nElems(), 124003);

    float *host_data;
    TMP::MemSpaceHost::allocate(&host_data, 124003);
    memo_space::copyToHost(host_data, vA.get_ptr(), 124003);
    for (size_t i(0); i < 124003; ++i)
        ASSERT_FLOAT_EQ(host_data[i], -1.0015);

    TMP::MemSpaceHost::release(host_data);
}