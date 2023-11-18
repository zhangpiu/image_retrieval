#include "image_retrieval/ann/binary_index.h"

#include <fstream>

#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "image_retrieval/feature_extraction/feature_decoder_utils.h"

namespace image_retrieval {
namespace ann {
namespace {

// 4 M
size_t BUFFER_SIZE{4 * 1024 * 1024};

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

template <int BitLength = 2048>
class BinaryIndex : public IndexBase {
 public:
  using FeatureRecord = ::image_retrieval::feature_extraction::FeatureRecord;

  BinaryIndex() : dim_size_(0), thread_pool_(10) {}

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
        bit_threshold_.resize(dim_size_, 0.f);
      }
      if (dim_size_ != record.value_size()) {
        throw std::runtime_error(absl::StrFormat(
            "Feature dim size should be equal, while got %d vs %ld", dim_size_,
            record.value_size()));
      }

      for (int i = 0; i < dim_size_; ++i) {
        bit_threshold_[i] += record.value(i);
      }
      index_data_[record.label()].emplace_back(std::move(record));

      if (++total_count_ % 1000 == 0) {
        std::cout << absl::StrFormat(
                         "Read %d records, elapsed %.3f(s)", total_count_,
                         (absl::ToUnixMicros(absl::Now()) - start) / 1e6)
                  << std::endl;
      }
    }

    if (total_count_) {
      for (int i = 0; i < dim_size_; ++i) {
        bit_threshold_[i] /= total_count_;
      }
    }

    for (const auto& kv : index_data_) {
      for (const auto& record : kv.second) {
        auto bits =
            Binarize(record.value().data(), bit_threshold_.data(), dim_size_);
        index_[record.label()].emplace_back(bits);
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

    auto query_bits = Binarize(query.data(), bit_threshold_.data(), dim_size_);
    std::vector<BucketRange> ranges;
    ranges.reserve(index_.size());
    size_t start = 0;
    for (const auto& kv : index_) {
      const auto& label = kv.first;
      if (request.labels.empty() || request.labels.count(label)) {
        ranges.push_back({label, start, start + index_[label].size()});
        start += index_[label].size();
      }
    }

    std::vector<RecordWithDistance> records(ranges.back().end);
    auto retrieve = [&](BucketRange range) {
      size_t index = range.start;
      size_t offset = 0;
      const auto& sub_index = index_[range.bucket];
      const auto& sub_index_data = index_data_[range.bucket];
      for (size_t i = 0; i < sub_index.size(); ++i) {
        float distance = (query_bits ^ sub_index[i]).count();
        records[index + offset].record = &sub_index_data[i];
        records[index + offset].distance = distance;
        ++offset;
      }
    };

    std::atomic_int join(ranges.size());
    for (const auto& range : ranges) {
      thread_pool_.Schedule([&]() {
        retrieve(range);
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
  std::bitset<BitLength> Binarize(const float* vector, const float* mean,
                                  int64_t length) const {
    std::bitset<BitLength> bits(length);
    for (int64_t i = 0; i < length; ++i) {
      if (vector[i] > mean[i]) {
        bits.set(i);
      }
    }
    return bits;
  }

 private:
  std::unordered_map<int, std::vector<std::bitset<BitLength>>> index_;

  std::unordered_map<int, std::vector<FeatureRecord>> index_data_;

  std::vector<float> bit_threshold_;

  int64_t dim_size_;

  concurrency::ThreadPool thread_pool_;
};
}  // namespace

template class ::image_retrieval::ann::BinaryIndex<2048>;

std::unique_ptr<IndexInterface> NewBinaryIndex2048() {
  return std::make_unique<BinaryIndex<2048>>();
}

}  // namespace ann
}  // namespace image_retrieval
