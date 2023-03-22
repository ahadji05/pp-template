#include "containers/Vector.hpp"
#include "test_defs.hpp"

TEST(test_constructor, GTest_Vector)
{
    TMP::Vector<float, size_t, memo_space> vA(10);

    std::cout << "Testing vector..." << std::endl;
    ASSERT_EQ(1, 1);
}