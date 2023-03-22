#include "containers/Vector.hpp"
#include "test_defs.hpp"

TEST(test_constructor, GTest_model)
{
    TMP::Vector<float, size_t, memo_space> vA(10);

    std::cout << "Testing model..." << std::endl;
    ASSERT_EQ(1, 1);
}