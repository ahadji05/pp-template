#include "containers/ScalarField.hpp"
#include "test_defs.hpp"

TEST(test_constructor_0, GTest_scalarField)
{
    // Check default contructor of ScalarField
    TMP::ScalarField<memo_space> a;
    ASSERT_EQ(a.get_nz(), 0);
    ASSERT_EQ(a.get_nx(), 0);
    ASSERT_EQ(a.get_nElems(), 0);
    ASSERT_EQ(a.get_nBytes(), 0);
}

TEST(test_constructor_1, GTest_scalarField)
{
    // Check contructor ScalarField(size_t nz, size_t nx)
    TMP::ScalarField<memo_space> a(6, 11);
    ASSERT_EQ(a.get_nz(), 6);
    ASSERT_EQ(a.get_nx(), 11);
    ASSERT_EQ(a.get_nElems(), 6 * 11);
    ASSERT_EQ(a.get_nBytes(), 6 * 11 * sizeof(float_type));
}

TEST(test_constructor_2, GTest_scalarField)
{
    // Check contructor ScalarField(size_t nz, size_t nx, vector_type othervector)
    float_type h_data[] = {float_type(1.0), float_type(2.0), float_type(2.0),
                           float_type(1.0), float_type(0.0), float_type(1.0)};
    TMP::Vector<float_type, memo_space> v(h_data, 6);

    TMP::ScalarField<memo_space> a(2, 3, v);
    ASSERT_EQ(a.get_nz(), 2);
    ASSERT_EQ(a.get_nx(), 3);
    for (size_t i(0); i < a.get_nElems(); ++i)
        ASSERT_EQ(a.get_ptr()[i], h_data[i]);
}

TEST(test_copy_constructor, GTest_scalarField)
{
    // Check copy-contructor ScalarField(const ScalarField &otherScalarField)
    float_type h_data[] = {float_type(1.0), float_type(2.0), float_type(2.0),
                           float_type(1.0), float_type(0.0), float_type(1.0)};
    TMP::Vector<float_type, memo_space> v(h_data, 6);
    TMP::ScalarField<memo_space> a(2, 3, v);

    TMP::ScalarField<memo_space> b(a);
    ASSERT_EQ(b.get_nz(), 2);
    ASSERT_EQ(b.get_nx(), 3);
    for (size_t i(0); i < b.get_nElems(); ++i)
        ASSERT_EQ(b.get_ptr()[i], h_data[i]);
}

TEST(test_copy_assignment, GTest_scalarField)
{
    // Check copy-assignment ScalarField(const ScalarField &otherScalarField)
    float_type h_data[] = {float_type(1.6),  float_type(2.2),   float_type(-0.53),
                           float_type(-1.1), float_type(-0.02), float_type(1.0034)};
    TMP::Vector<float_type, memo_space> v(h_data, 5);
    TMP::ScalarField<memo_space> a(1, 5, v);

    TMP::ScalarField<memo_space> b = a;
    ASSERT_EQ(b.get_nz(), 1);
    ASSERT_EQ(b.get_nx(), 5);
    for (size_t i(0); i < b.get_nElems(); ++i)
        ASSERT_EQ(b.get_ptr()[i], h_data[i]);
}

TEST(test_copy_assignment_reuse, GTest_scalarField)
{
    // Check copy-assignment ScalarField(const ScalarField &otherScalarField)
    // internally selects the path that avoids deep-copy
    float_type h_data[] = {float_type(1.6),  float_type(2.2),   float_type(-0.53),
                           float_type(-1.1), float_type(-0.02), float_type(1.0034)};
    TMP::Vector<float_type, memo_space> v(h_data, 5);
    TMP::ScalarField<memo_space> a(1, 5, v);
    TMP::ScalarField<memo_space> b(5, 1);

    b = a; // b has already an allocated vector with suitable size
    ASSERT_EQ(b.get_nz(), 1);
    ASSERT_EQ(b.get_nx(), 5);
    for (size_t i(0); i < b.get_nElems(); ++i)
        ASSERT_EQ(b.get_ptr()[i], h_data[i]);
}

TEST(test_destructor, GTest_scalarField)
{
    // Check destructor ~ScalarField()
    float_type h_data[] = {float_type(1.6),  float_type(2.2),   float_type(-0.53),
                           float_type(-1.1), float_type(-0.02), float_type(1.0034)};
    TMP::Vector<float_type, memo_space> v(h_data, 5);
    TMP::ScalarField<memo_space> a(1, 5, v);

    TMP::ScalarField<memo_space> *b = new TMP::ScalarField<memo_space>(a);

    ASSERT_EQ(b->get_nz(), 1);
    ASSERT_EQ(b->get_nx(), 5);
    ASSERT_EQ(b->get_nElems(), 5);
    ASSERT_EQ(b->get_nBytes(), 5 * sizeof(float_type));
    for (size_t i(0); i < b->get_nElems(); ++i)
        ASSERT_EQ(b->get_ptr()[i], h_data[i]);

    ASSERT_NO_THROW(delete b);
}