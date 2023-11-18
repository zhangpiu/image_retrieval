#include "image_retrieval/ann/flat_index.h"

#include <algorithm>
#include <fstream>

#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "image_retrieval/ann/vector_distance.h"
#include "image_retrieval/feature_extraction/feature_decoder_utils.h"

namespace image_retrieval {
namespace ann {
namespace {

// 4 M
size_t BUFFER_SIZE{4 * 1024 * 1024};

using ::image_retrieval::concurrency::ThreadPool;
using ::image_retrieval::feature_extraction::FeatureRecord;
using ::image_retrieval::feature_extraction::ReadRecord;

struct RecordWithDistance {
  explicit RecordWithDistance(const FeatureRecord* record = nullptr,
                              float distance = 0.f)
      : record(record), distance(distance) {}

  const FeatureRecord* record;
  float distance;
};

struct BucketRange {
  int bucket;
  size_t start;
  size_t end;
};

class FlatIndex : public IndexBase {
 public:
  FlatIndex() : dim_size_(0), thread_pool_(10) {}

  using FeatureRecord = ::image_retrieval::feature_extraction::FeatureRecord;

  bool BuildIndex(const std::string& filepath) override {
    std::ifstream file(filepath);
    if (!file.good()) {
      throw std::runtime_error(absl::StrFormat(
          "%s does not exist, please investigate and retry!", filepath));
    }

    int64_t start = absl::ToUnixMicros(absl::Now());
    std::vector<char> buffer(BUFFER_SIZE);
    while (true) {
      if (!ReadRecord(file, buffer)) {
        break;
      }

      FeatureRecord record;
      record.ParseFromArray(buffer.data(), buffer.size());
      if (dim_size_ == 0) {
        dim_size_ = record.value_size();
      } else {
        if (dim_size_ != record.value_size()) {
          throw std::runtime_error(absl::StrFormat(
              "Feature dim size should be equal, while got %d vs %ld",
              dim_size_, record.value_size()));
        }
      }

      index_[record.label()].emplace_back(record);
      if (++total_count_ % 1000 == 0) {
        std::cout << absl::StrFormat(
                         "Read %d records, elapsed %.3f(s)", total_count_,
                         (absl::ToUnixMicros(absl::Now()) - start) / 1e6)
                  << std::endl;
      }
    }

    std::cout << absl::StrFormat(
                     "Totally read %d records, elapsed %.3f(s)", total_count_,
                     (absl::ToUnixMicros(absl::Now()) - start) / 1e6)
              << std::endl;

    return true;
  }

  bool Search(const SearchRequest& request, SearchResponse& response) override {
    if (index_.empty()) {
      return true;
    }

    const auto& query = request.query;
    if (query.size() != dim_size_) {
      throw std::runtime_error(
          absl::StrFormat("Query feature dim size should be equal to index "
                          "feature, while got %d vs %d",
                          query.size(), dim_size_));
    }

    std::vector<BucketRange> ranges;
    ranges.reserve(index_.size());
    size_t start = 0;
    for (const auto& kv : index_) {
      int label = kv.first;
      if (request.labels.empty() || request.labels.count(label)) {
        size_t offset = index_[label].size();
        ranges.push_back({label, start, start + offset});
        start += offset;
      }
    }
    if (ranges.empty()) {
      return true;
    }

    size_t search_count = ranges.back().end;
    std::vector<RecordWithDistance> records(search_count);
    auto retrieve_fn = [&](BucketRange range) {
      size_t index = range.start;
      size_t offset = 0;
      for (const auto& record : index_[range.bucket]) {
        float distance = Avx256CosineDistance(
            query.data(), record.value().data(), query.size());
        records[index + offset].record = &record;
        records[index + offset].distance = distance;
        ++offset;
      }
    };

    std::atomic_int join(ranges.size());
    for (BucketRange range : ranges) {
      thread_pool_.Schedule([&, range]() {
        retrieve_fn(range);
        --join;
      });
    }

    while (join) {
    }

    int partial_size =
        request.top_k < records.size() ? request.top_k : records.size();
    std::partial_sort(
        records.begin(), records.begin() + partial_size, records.end(),
        [](const RecordWithDistance& x, const RecordWithDistance& y) {
          return x.distance < y.distance;
        });

    if (records.size() > request.top_k) {
      records.resize(request.top_k);
    }

    response.total_count = total_count_;
    auto* neighbors = &response.neighbors;
    for (const auto& record : records) {
      neighbors->emplace_back();
      auto* response_record = &neighbors->back();
      response_record->CopyFrom(*record.record);
      response_record->distance = record.distance;
    }

    return true;
  }

 private:
  std::unordered_map<int, std::vector<FeatureRecord>> index_;

  int64_t dim_size_;

  concurrency::ThreadPool thread_pool_;
};

}  // namespace

std::unique_ptr<IndexInterface> NewFlatIndex() {
  return std::make_unique<FlatIndex>();
}

}  // namespace ann
}  // namespace image_retrieval
