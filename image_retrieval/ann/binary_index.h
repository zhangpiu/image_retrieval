//
// Created by david on 2021/7/1.
//

#ifndef IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_ANN_BINARY_INDEX_H_
#define IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_ANN_BINARY_INDEX_H_

#include <bitset>
#include <unordered_map>
#include "image_retrieval/ann/index_interface.h"
#include "image_retrieval/concurrency/thread_pool.h"

namespace image_retrieval {
namespace ann {

std::unique_ptr<IndexInterface> NewBinaryIndex2048();

}  // namespace ann
}  // namespace image_retrieval

#endif  // IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_ANN_BINARY_INDEX_H_
