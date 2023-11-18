#include "image_retrieval/clustering/kmeans.h"

#include <atomic>
#include <iostream>
#include <numeric>
#include <random>
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"

namespace image_retrieval {
namespace clustering {

std::vector<int> Permutation(size_t max) {
  std::vector<int> perm(max);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), std::mt19937(std::random_device()()));
  return perm;
}

int NearestCentroid(const Eigen::MatrixXf& centroids,
                    const Eigen::RowVectorXf& point) {
  int index;
  Eigen::MatrixXf diff = centroids.rowwise() - point;
  diff.rowwise().squaredNorm().minCoeff(&index);
  return index;
}

void KMeans::Train(const Eigen::MatrixXf& data,
                   const Eigen::MatrixXf& init_centroids,
                   const std::vector<int>& init_membership) {
  if (data.rows() < k_) {
    throw std::runtime_error(
        absl::StrFormat("Number of training points (%ld) should be at least as "
                        "large as number of clusters (%d)",
                        data.rows(), k_));
  }

  long n = data.rows();
  long dimension = data.cols();
  if (init_centroids.size()) {
    if (init_centroids.rows() != k_ || init_centroids.cols() != data.cols()) {
      throw std::runtime_error(absl::StrFormat(
          "Size of initial centroids should be (%d, %ld), while got (%ld, %ld)",
          k_, data.cols(), init_centroids.rows(), init_centroids.cols()));
    }
    centroids_ = init_centroids;
  } else {
    centroids_.resize(k_, data.cols());
    std::vector<int> perm = Permutation(n);
    for (long i = 0; i < k_; ++i) {
      centroids_.row(i) = data.row(perm[i]);
    }
  }

  // Initialization: assigning all data points to a dummy cluster
  membership_ = Eigen::MatrixXi::Constant(1, n, k_);
  int64_t start = absl::ToUnixMicros(absl::Now());

  for (int it = 0; it < max_iteration_; ++it) {
    Eigen::MatrixXf centroids = Eigen::MatrixXf::Zero(k_, dimension);
    std::vector<int> counter(k_, 0);
    std::atomic_long assign(0);

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < data.rows(); ++i) {
      int centroid_index = NearestCentroid(centroids_, data.row(i));
      if (membership_[i] != centroid_index) {
        membership_[i] = centroid_index;
        ++assign;
      }
    }

    for (int i = 0; i < data.rows(); ++i) {
      int centroid_index = membership_[i];
      centroids.row(centroid_index) =
          counter[centroid_index] * centroids.row(centroid_index) + data.row(i);
      // To prevent overflow
      centroids.row(centroid_index) /= ++counter[centroid_index];
    }

    std::cout << absl::StrFormat(
                     "Iteration %d: %ld points reassigned, elapsed %.3f(s)", it,
                     assign.load(),
                     (absl::ToUnixMicros(absl::Now()) - start) / 1e6)
              << std::endl;
    if (!frozen_centroids_ && assign > 0) {
      centroids_.swap(centroids);
    }
  }
}

}  // namespace clustering
}  // namespace image_retrieval