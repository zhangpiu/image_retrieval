#ifndef IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_CLUSTERING_KMEANS_H_
#define IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_CLUSTERING_KMEANS_H_

#include <vector>
#include "eigen3/Eigen/Dense"

namespace image_retrieval {
namespace clustering {

class KMeans {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  KMeans(int k, int max_iteration, bool frozen_centroids = false)
      : k_(k),
        max_iteration_(max_iteration),
        frozen_centroids_(frozen_centroids) {}

  KMeans(const KMeans&) = delete;

  KMeans& operator=(const KMeans&) = delete;

  void Train(const Eigen::MatrixXf& data,
             const Eigen::MatrixXf& init_centroids = Eigen::MatrixXf(),
             const std::vector<int>& init_membership = {});

  const Eigen::MatrixXf& GetCentroids() const { return centroids_; }

  const Eigen::RowVectorXi GetMembership() const { return membership_; }

 private:
  int k_;

  int max_iteration_;

  bool frozen_centroids_;

  Eigen::MatrixXf centroids_;

  Eigen::RowVectorXi membership_;
};

}  // namespace clustering
}  // namespace image_retrieval

#endif  // IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_CLUSTERING_KMEANS_H_
