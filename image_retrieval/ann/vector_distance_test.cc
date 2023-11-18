#include "image_retrieval/ann/vector_distance.h"

#include <vector>
#include "absl/random/random.h"
#include "gtest/gtest.h"

namespace image_retrieval {
namespace ann {
namespace {


void TestCosineDistance(size_t dim = 32) {
  std::vector<float> x(dim, 0), y(dim, 0);
  absl::BitGen bit_gen;
  for (size_t i = 0; i < dim; ++i) {
    x[i] = absl::Uniform<float>(bit_gen, .1f, 1.f);
    y[i] = absl::Uniform<float>(bit_gen, -1.f, 1.f);
  }

  float d1 = BaselineCosineDistance(x.data(), y.data(), x.size());

#if defined(_ENABLE_AVX) && defined(__AVX__)
  float d2 = Avx256CosineDistance(x.data(), y.data(), x.size());
  EXPECT_FLOAT_EQ(d1, d2);
#else
  static_assert(false, "AVX is not available, please check and recompile!");
#endif
}

TEST(AVX, Basic) {
  TestCosineDistance(8);
  TestCosineDistance(16);
  TestCosineDistance(2048);
}

}  // namespace
}  // namespace ann
}  // namespace image_retrieval
