#ifndef IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_ANN_INDEX_INTERFACE_H_
#define IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_ANN_INDEX_INTERFACE_H_

#include <unordered_set>
#include "google/protobuf/util/json_util.h"
#include "image_retrieval/feature_extraction/feature.pb.h"
#include "nlohmann/json.hpp"

namespace image_retrieval {
namespace ann {

struct SearchRequest {
  std::vector<float> query;
  int top_k = 20;
  std::unordered_set<int> labels;

  friend void to_json(nlohmann::json& j, const SearchRequest& request) {
    j = nlohmann::json{{"query", request.query},
                       {"top_k", request.top_k},
                       {"labels", request.labels}};
  }

  friend void from_json(const nlohmann::json& j, SearchRequest& request) {
    request.query = j.at("query").get<std::vector<float>>();
    if (j.contains("top_k")) {
      request.top_k = j.at("top_k").get<int>();
    }
    if (j.contains("labels")) {
      request.labels = j.at("labels").get<std::unordered_set<int>>();
    }
  }
};

struct ResponseRecord : public feature_extraction::FeatureRecord {
  float distance;

  friend void to_json(nlohmann::json& j, const ResponseRecord& record) {
    std::string output;
    auto option = google::protobuf::util::JsonOptions();
    option.add_whitespace = true;
    option.preserve_proto_field_names = true;
    google::protobuf::util::MessageToJsonString(record, &output, option);

    j = nlohmann::json::parse(output);
    j["distance"] = record.distance;
  }
};

struct SearchResponse {
  std::vector<ResponseRecord> neighbors;
  float search_cost_ms;
  int64_t total_count;

  friend void to_json(nlohmann::json& j, const SearchResponse& response) {
    j = nlohmann::json{{"neighbors", response.neighbors},
                       {"search_cost_ms", response.search_cost_ms},
                       {"total_count", response.total_count}};
  }
};

class IndexInterface {
 public:
  virtual ~IndexInterface() = default;

  virtual bool BuildIndex(const std::string& filepath) = 0;

  virtual bool Search(const SearchRequest& request,
                      SearchResponse& response) = 0;
};

class IndexBase : public IndexInterface {
 public:
  IndexBase() : total_count_(0) {}

 protected:
  int64_t total_count_;
};

}  // namespace ann
}  // namespace image_retrieval

#endif  // IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_ANN_INDEX_INTERFACE_H_
