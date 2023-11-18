#include <iostream>

#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "cmdline/cmdline.h"
#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"

#include "image_retrieval/ann/binary_index.h"
#include "image_retrieval/ann/flat_index.h"

using ::image_retrieval::ann::IndexInterface;
using ::image_retrieval::ann::NewBinaryIndex2048;
using ::image_retrieval::ann::NewFlatIndex;
using ::image_retrieval::ann::SearchRequest;
using ::image_retrieval::ann::SearchResponse;

int main(int argc, char* argv[]) {
  std::ios::sync_with_stdio(false);
  cmdline::parser parser;
  parser.add<std::string>("input", 'i', "Input filename", true, "");
  parser.add<std::string>("index_type", 't', "Index type, 'flat' or 'binary'",
                          false, "flat",
                          cmdline::oneof<std::string>("flat", "binary"));
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

  std::unique_ptr<IndexInterface> index;
  if (index_type == "flat") {
    index = NewFlatIndex();
  } else {
    index = NewBinaryIndex2048();
  }
  index->BuildIndex(filename);

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
