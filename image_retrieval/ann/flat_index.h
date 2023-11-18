#ifndef IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_ANN_FLAT_INDEX_H_
#define IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_ANN_FLAT_INDEX_H_

#include <unordered_map>
#include "image_retrieval/ann/index_interface.h"
#include "image_retrieval/concurrency/thread_pool.h"

namespace image_retrieval {
namespace ann {

std::unique_ptr<IndexInterface> NewFlatIndex();

}  // namespace ann
}  // namespace image_retrieval
#endif  // IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_ANN_FLAT_INDEX_H_
