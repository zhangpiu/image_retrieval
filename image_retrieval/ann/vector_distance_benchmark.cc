#include "image_retrieval/ann/vector_distance.h"

#include <vector>
#include "absl/random/random.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

namespace image_retrieval {
namespace ann {
namespace {

void BM_CosineDistance(benchmark::State& state) {  // NOLINT
  size_t dim = state.range(0);
  std::vector<float> x(dim, 0), y(dim, 0);
  absl::BitGen bit_gen;
  for (size_t i = 0; i < dim; ++i) {
    x[i] = absl::Uniform<float>(bit_gen, .1f, 1.f);
    y[i] = absl::Uniform<float>(bit_gen, -1.f, 1.f);
  }

  for (auto _ : state) {
    BaselineCosineDistance(x.data(), y.data(), x.size());
  }
}

void BM_AvxCosineDistance(benchmark::State& state) {  // NOLINT
  size_t dim = state.range(0);
  std::vector<float> x(dim, 0), y(dim, 0);
  absl::BitGen bit_gen;
  for (size_t i = 0; i < dim; ++i) {
    x[i] = absl::Uniform<float>(bit_gen, .1f, 1.f);
    y[i] = absl::Uniform<float>(bit_gen, -1.f, 1.f);
  }

#if defined(_ENABLE_AVX) && defined(__AVX__)
  for (auto _ : state) {
    Avx256CosineDistance(x.data(), y.data(), x.size());
  }
#else
  static_assert(false, "AVX is not available, please check and recompile!");
#endif
}

BENCHMARK(BM_CosineDistance)->Arg(16)->Arg(64)->Arg(2048);
BENCHMARK(BM_AvxCosineDistance)->Arg(16)->Arg(64)->Arg(2048);

}  // namespace
}  // namespace ann
}  // namespace image_retrieval

BENCHMARK_MAIN();
