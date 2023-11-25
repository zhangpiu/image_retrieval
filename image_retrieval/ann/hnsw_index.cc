#include "image_retrieval/ann/flat_index.h"

#include <algorithm>
#include <fstream>

#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "image_retrieval/feature_extraction/feature.pb.h"
#include "third_party/hnswlib/hnswlib.h"

namespace image_retrieval {
namespace ann {
namespace {

using ::image_retrieval::concurrency::ThreadPool;
using ::image_retrieval::feature_extraction::FeatureRecord;

class HNSWIndex : public IndexBase {
 public:
  explicit HNSWIndex(int dim_size) : IndexBase(dim_size) {
    space_ = std::make_unique<hnswlib::L2Space>(dim_size_);
    alg_hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space_.get(), max_elements_, M_, ef_construction_);
  }

  using FeatureRecord = ::image_retrieval::feature_extraction::FeatureRecord;

  bool Add(const FeatureRecord& record) override {
    alg_hnsw_->addPoint(record.value().data(), index_.size());
    index_.push_back(record);
    ++total_count_;

    return true;
  }

  bool Search(const SearchRequest& request, SearchResponse& response) override {
    const auto& query = request.query;
    if (query.size() != dim_size_) {
      throw std::runtime_error(
          absl::StrFormat("Query feature dim size should be equal to index "
                          "feature, while got %d vs %d",
                          query.size(), dim_size_));
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
        alg_hnsw_->searchKnn(query.data(), request.top_k);

    response.total_count = total_count_;
    auto* neighbors = &response.neighbors;
    neighbors->resize(result.size());

    while (!result.empty()) {
      std::pair<float, hnswlib::labeltype> element = result.top();
      result.pop();
      ResponseRecord* response_record = &neighbors->at(result.size());
      response_record->CopyFrom(index_[element.second]);
      response_record->distance = element.first;
    }

    return true;
  }

 private:
  // Maximum number of elements, should be known beforehand
  int max_elements_ = 1300000;

  // Tightly connected with internal dimensionality of the data
  int M_ = 16;

  // Controls index search speed/build speed tradeoff
  int ef_construction_ = 200;

  std::unique_ptr<hnswlib::L2Space> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw_;
  std::vector<FeatureRecord> index_;
};

}  // namespace

std::unique_ptr<IndexInterface> NewHNSWIndex(int dim_size) {
  return std::make_unique<HNSWIndex>(dim_size);
}

}  // namespace ann
}  // namespace image_retrieval
