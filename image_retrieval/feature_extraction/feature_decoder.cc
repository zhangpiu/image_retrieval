#include <fstream>
#include <iostream>
#include "cmdline/cmdline.h"
#include "feature_decoder_utils.h"
#include "image_retrieval/feature_extraction/feature.pb.h"
#include "google/protobuf/util/json_util.h"

using image_retrieval::feature_extraction::FeatureRecord;
using image_retrieval::feature_extraction::ReadRecord;

// 4 M
size_t BUFFER_SIZE{4 * 1024 * 1024};

int main(int argc, char* argv[]) {
  std::ios::sync_with_stdio(false);
  cmdline::parser parser;
  parser.add<std::string>(
      "input", 'i', "Input filename(if empty, read from stdin)", false, "");
  parser.add<std::string>("format", 'f', "Output format(default, json)", false,
                          "default",
                          cmdline::oneof<std::string>("default", "json"));
  parser.add<size_t>("limit", 'l', "Test limit", false,
                     std::numeric_limits<size_t>::max());
  parser.add("help", 0, "print this message");
  bool ok = parser.parse(argc, argv);
  if (not ok) {
    std::cerr << parser.usage();
    return 1;
  }

  const auto& filename = parser.get<std::string>("input");
  const auto& format = parser.get<std::string>("format");
  const auto& limit = parser.get<size_t>("limit");

  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (not filename.empty() and not file.good()) {
    std::cerr << filename
              << ": does not exist, please investigate and retry!\n";
    return 1;
  }

  std::vector<char> buffer(BUFFER_SIZE);
  for (size_t i = 0; i < limit; ++i) {
    if (!ReadRecord(filename.empty() ? std::cin : file, buffer)) {
      break;
    }

    FeatureRecord feature_record;
    feature_record.ParseFromArray(buffer.data(), buffer.size());

    std::string line(40, '-');
    ::fprintf(stdout, "%s #%06luth record %s\n", line.c_str(), i + 1,
              line.c_str());
    if (format == "default") {
      std::cout << feature_record.DebugString() << std::endl;
    } else if (format == "json") {
      std::string output;
      auto option = google::protobuf::util::JsonOptions();
      option.add_whitespace = true;
      option.preserve_proto_field_names = true;
      google::protobuf::util::MessageToJsonString(feature_record, &output,
                                                  option);
      std::cout << output << std::endl;
    } else {
      assert(false);
    }
  }
}
