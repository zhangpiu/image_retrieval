#ifndef IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_ANN_VECTOR_DISTANCE_H_
#define IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_ANN_VECTOR_DISTANCE_H_

#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>

#if defined(_ENABLE_AVX) && defined(__AVX__)
#include <immintrin.h>
#endif

namespace image_retrieval {
namespace ann {

template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
IsAlmostEqual(T x, T y, int ulp = 2) {
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  return std::abs(x - y) <=
             std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp
         // unless the result is subnormal
         || std::abs(x - y) < std::numeric_limits<T>::min();
}

inline float BaselineCosineDistance(const float* x, const float* y,
                                    int64_t length) {
  float dot = 0.f, norm_x = 0.f, norm_y = 0.f;
  for (int64_t i = 0; i < length; ++i) {
    dot += x[i] * y[i];
    norm_x += x[i] * x[i];
    norm_y += y[i] * y[i];
  }

  if (IsAlmostEqual(norm_x, 0.f) || IsAlmostEqual(norm_y, 0.f)) {
    return 1.f;
  }

  return 1.f - dot / std::sqrt(norm_x * norm_y);
}

inline float BaselineEuclideanDistance(const float* x, const float* y,
                                       int64_t length) {
  float distance = 0.f;
  for (int64_t i = 0; i < length; ++i) {
    distance += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return distance;
}

#if defined(_ENABLE_AVX) && defined(__AVX__)
inline float ReduceM128(__m128 num) {
  __attribute__((aligned(16))) float f[4] = {0.f};
  _mm_store_ps(f, num);
  return f[0] + f[1] + f[2] + f[3];
}

inline float ReduceM256(__m256 num) {
  __m128 hi = _mm256_extractf128_ps(num, 1);
  __m128 lo = _mm256_extractf128_ps(num, 0);
  hi = _mm_add_ps(hi, lo);
  return ReduceM128(hi);
}

inline float Avx256CosineDistance(const float* x, const float* y,
                                  int64_t length) {
  assert(length % 8 == 0);

  // OPTIMIZE: Loads floating-point vector from an aligned memory address
  __m256 _dot = _mm256_setzero_ps();
  __m256 _norm_x = _mm256_setzero_ps();
  __m256 _norm_y = _mm256_setzero_ps();
  for (; length > 7; length -= 8, x += 8, y += 8) {
    const __m256 _x = _mm256_loadu_ps(x);
    const __m256 _y = _mm256_loadu_ps(y);
    _dot = _mm256_fmadd_ps(_x, _y, _dot);
    _norm_x = _mm256_fmadd_ps(_x, _x, _norm_x);
    _norm_y = _mm256_fmadd_ps(_y, _y, _norm_y);
  }

  float dot = ReduceM256(_dot);
  float norm_x = ReduceM256(_norm_x);
  float norm_y = ReduceM256(_norm_y);

  if (IsAlmostEqual(norm_x, 0.f) || IsAlmostEqual(norm_y, 0.f)) {
    return 1.f;
  }

  return 1.f - dot / std::sqrt(norm_x * norm_y);
}
#endif

}  // namespace ann
}  // namespace image_retrieval

#endif  // IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_ANN_VECTOR_DISTANCE_H_
