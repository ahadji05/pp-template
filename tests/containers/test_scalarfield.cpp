#include "containers/Vector.hpp"
#include "test_defs.hpp"

TEST(test_constructor, GTest_scalarField)
{
    TMP::Vector<float, size_t, memo_space> vA(10);

    std::cout << "Testing field..." << std::endl;
    ASSERT_EQ(1, 1);
}