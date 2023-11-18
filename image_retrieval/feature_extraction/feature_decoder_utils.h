#ifndef IMAGE_RETRIEVAL_FEATURE_EXTRACTION_FEATURE_DECODER_UTILS_H_
#define IMAGE_RETRIEVAL_FEATURE_EXTRACTION_FEATURE_DECODER_UTILS_H_

#include <istream>
#include <vector>

namespace image_retrieval {
namespace feature_extraction {

static bool ReadRecord(std::istream& input, std::vector<char>& buffer) {
  if (input.eof()) {
    return false;
  }

  uint64_t size = 0;
  bool result =
      (bool)(input.read(reinterpret_cast<char*>(&size), sizeof(size)));
  if (input.eof() || input.fail() || !result) {
    return false;
  }

  if (size >= buffer.capacity()) {
    std::cerr << "Not enough buffer size while reading bytes, size=" << size
              << " buffer_size=" << buffer.size() << std::endl;
    return false;
  }

  buffer.resize(size);
  if (!input.read(buffer.data(), size)) {
    std::cerr << "Unexpected error while reading bytes, size=" << size
              << " buffer_size=" << buffer.size() << std::endl;
    return false;
  }

  return true;
}

}  // namespace feature_extraction
}  // namespace image_retrieval

#endif  // IMAGE_RETRIEVAL_FEATURE_EXTRACTION_FEATURE_DECODER_UTILS_H_
