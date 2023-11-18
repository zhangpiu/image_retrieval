#include "image_retrieval/clustering/kmeans.h"

#include "gtest/gtest.h"

namespace image_retrieval {
namespace clustering {
namespace {

TEST(KMeans, Basic) {
  KMeans kmeans(2, 20);
  Eigen::MatrixXf data(3, 4);
  data << 1, 1, 1, 1, 2, 2, 2, 2, 8, 8, 8, 8;
  kmeans.Train(data);
  Eigen::MatrixXf centroids = kmeans.GetCentroids();
  Eigen::MatrixXi membership = kmeans.GetMembership();

  std::cout << centroids << std::endl;
  std::cout << membership << std::endl;
}

}  // namespace
}  // namespace clustering
}  // namespace image_retrieval