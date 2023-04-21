#include "ppt/containers/ScalarField.hpp"
#include "test_defs.hpp"

TEST(test_constructor_0, GTest_scalarField) {
  // Check default contructor of ScalarField
  ScalarField<memo_space> a;
  ASSERT_EQ(a.get_nz(), 0);
  ASSERT_EQ(a.get_nx(), 0);
  ASSERT_EQ(a.get_nElems(), 0);
}

TEST(test_constructor_1, GTest_scalarField) {
  // Check contructor ScalarField(size_t nz, size_t nx)
  ScalarField<memo_space> a(6, 11);
  ASSERT_EQ(a.get_nz(), 6);
  ASSERT_EQ(a.get_nx(), 11);
  ASSERT_EQ(a.get_nElems(), 6 * 11);
}

TEST(test_constructor_2_explicit_host_container, GTest_scalarField) {
  // Check contructor ScalarField(size_t nz, size_t nx, vector_type othervector)
  float_type h_data[] = {float_type(1.0), float_type(2.0), float_type(2.0),
                         float_type(1.0), float_type(0.0), float_type(1.0)};
  typename ScalarField<TMP::MemSpaceHost>::vector_type v(h_data, 6);

  ScalarField<TMP::MemSpaceHost> a(2, 3, v);
  ASSERT_EQ(a.get_nz(), 2);
  ASSERT_EQ(a.get_nx(), 3);
  for (size_t i(0); i < a.get_nElems(); ++i)
    ASSERT_FLOAT_EQ(a.get_ptr()[i], h_data[i]);
}

TEST(test_copy_constructor, GTest_scalarField) {
  // Check copy-contructor ScalarField(const ScalarField &otherScalarField)
  float_type h_data[] = {float_type(1.0), float_type(2.0), float_type(2.0),
                         float_type(1.0), float_type(0.0), float_type(1.0)};
  typename ScalarField<memo_space>::vector_type v(h_data, 6);
  ScalarField<memo_space> a(2, 3, v);

  ScalarField<memo_space> b(a);
  ASSERT_EQ(b.get_nz(), 2);
  ASSERT_EQ(b.get_nx(), 3);

  float_type *host_data;
  TMP::MemSpaceHost::allocate(&host_data, 6);
  memo_space::copyToHost(host_data, b.get_ptr(), 6);
  for (size_t i(0); i < b.get_nElems(); ++i) ASSERT_EQ(host_data[i], h_data[i]);

  TMP::MemSpaceHost::release(host_data);
}

TEST(test_copy_assignment, GTest_scalarField) {
  // Check copy-assignment ScalarField(const ScalarField &otherScalarField)
  float_type h_data[] = {float_type(1.6),   float_type(2.2),
                         float_type(-0.53), float_type(-1.1),
                         float_type(-0.02), float_type(1.0034)};
  typename ScalarField<memo_space>::vector_type v(h_data, 6);
  ScalarField<memo_space> a(1, 6, v);

  ScalarField<memo_space> b = a;
  ASSERT_EQ(b.get_nz(), 1);
  ASSERT_EQ(b.get_nx(), 6);

  float_type *host_data;
  TMP::MemSpaceHost::allocate(&host_data, 6);
  memo_space::copyToHost(host_data, b.get_ptr(), 6);
  for (size_t i(0); i < b.get_nElems(); ++i) ASSERT_EQ(host_data[i], h_data[i]);

  TMP::MemSpaceHost::release(host_data);
}

TEST(test_destructor, GTest_scalarField) {
  // Check destructor ~ScalarField()
  float_type h_data[] = {float_type(1.6),   float_type(2.2),
                         float_type(-0.53), float_type(-1.1),
                         float_type(-0.02), float_type(1.0034)};
  typename ScalarField<memo_space>::vector_type v(h_data, 6);
  ScalarField<memo_space> a(1, 6, v);

  ScalarField<memo_space> *b = new ScalarField<memo_space>(a);

  ASSERT_EQ(b->get_nz(), 1);
  ASSERT_EQ(b->get_nx(), 6);
  ASSERT_EQ(b->get_nElems(), 6);
  ASSERT_NO_THROW(delete b);
}