#include <iostream>

#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "cmdline/cmdline.h"
#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"

#include "image_retrieval/ann/binary_index.h"
#include "image_retrieval/ann/flat_index.h"
#include "image_retrieval/ann/hnsw_index.h"
#include "image_retrieval/feature_extraction/feature_decoder_utils.h"

using ::image_retrieval::ann::IndexInterface;
using ::image_retrieval::ann::NewBinaryIndex2048;
using ::image_retrieval::ann::NewFlatIndex;
using ::image_retrieval::ann::NewHNSWIndex;
using ::image_retrieval::ann::SearchRequest;
using ::image_retrieval::ann::SearchResponse;
using ::image_retrieval::feature_extraction::FeatureRecord;
using ::image_retrieval::feature_extraction::ReadRecord;

// 4 M
size_t BUFFER_SIZE{4 * 1024 * 1024};

bool BuildIndex(const std::string& filepath, IndexInterface* index) {
  std::ifstream file(filepath);
  if (!file.good()) {
    throw std::runtime_error(absl::StrFormat(
        "%s does not exist, please investigate and retry!", filepath));
  }

  int dim_size = index->GetDimSize();
  int64_t total_count = 0;
  int64_t start = absl::ToUnixMicros(absl::Now());
  std::vector<char> buffer(BUFFER_SIZE);
  while (true) {
    if (!ReadRecord(file, buffer)) {
      break;
    }

    FeatureRecord record;
    record.ParseFromArray(buffer.data(), buffer.size());
    if (dim_size != record.value_size()) {
      throw std::runtime_error(absl::StrFormat(
          "Feature dim size should be equal, while got %d vs %ld", dim_size,
          record.value_size()));
    }

    index->Add(record);
    if (++total_count % 1000 == 0) {
      std::cout << absl::StrFormat(
                       "Read %d records, elapsed %.3f(s)", total_count,
                       (absl::ToUnixMicros(absl::Now()) - start) / 1e6)
                << std::endl;
    }
  }

  std::cout << absl::StrFormat("Totally read %d records, elapsed %.3f(s)",
                               total_count,
                               (absl::ToUnixMicros(absl::Now()) - start) / 1e6)
            << std::endl;

  return true;
}

int main(int argc, char* argv[]) {
  std::ios::sync_with_stdio(false);
  cmdline::parser parser;
  parser.add<std::string>("input", 'i', "Input filename", true, "");
  parser.add<std::string>(
      "index_type", 't', "Index type, 'flat' or 'binary' or 'hnsw'", false,
      "flat", cmdline::oneof<std::string>("flat", "binary", "hnsw"));
  parser.add<int>("dim", 'd', "Dimension size of feature", false, 2048);
  parser.add<int>("port", 'p', "port number", false, 8080,
                  cmdline::range(1, 65535));
  parser.add("help", 0, "print this message");
  bool ok = parser.parse(argc, argv);
  if (not ok) {
    std::cerr << parser.usage();
    return 1;
  }

  const auto& filename = parser.get<std::string>("input");
  const auto& index_type = parser.get<std::string>("index_type");
  int port = parser.get<int>("port");
  int dim_size = parser.get<int>("dim");

  std::unique_ptr<IndexInterface> index;
  if (index_type == "flat") {
    index = NewFlatIndex(dim_size);
  } else if (index_type == "binary") {
    if (dim_size != 2048) {
      throw std::invalid_argument(
          "Binary index only supports dim_size=2048 yet.");
    }
    index = NewBinaryIndex2048(dim_size);
  } else {
    index = NewHNSWIndex(dim_size);
  }
  BuildIndex(filename, index.get());

  httplib::Server server;
  server.Post(R"(/search)", [&](const httplib::Request& request,
                                httplib::Response& response) {
    SearchRequest search_request;
    try {
      nlohmann::json json = nlohmann::json::parse(request.body);
      search_request = json.get<SearchRequest>();
    } catch (const std::exception& e) {
      response.set_content(absl::StrFormat("Bad request: %s\n", e.what()),
                           "text/plain");
      return;
    }

    try {
      SearchResponse search_response;
      int64_t start = absl::ToUnixMicros(absl::Now());
      index->Search(search_request, search_response);
      int64_t search_cost = absl::ToUnixMicros(absl::Now()) - start;
      search_response.search_cost_ms = search_cost / 1000.f;

      nlohmann::json output = search_response;
      response.set_content(output.dump(2), "text/plain");
    } catch (const std::exception& e) {
      response.set_content(absl::StrFormat("Internal error: %s\n", e.what()),
                           "text/plain");
    }
  });

  server.listen("0.0.0.0", port);

  return 0;
}
