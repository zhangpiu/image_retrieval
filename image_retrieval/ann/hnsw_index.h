#ifndef IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_ANN_HNSW_INDEX_H_
#define IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_ANN_HNSW_INDEX_H_

#include "image_retrieval/ann/index_interface.h"

namespace image_retrieval {
namespace ann {

std::unique_ptr<IndexInterface> NewHNSWIndex(int dim_size);

}  // namespace ann
}  // namespace image_retrieval

#endif  // IMAGE_RETRIEVAL_IMAGE_RETRIEVAL_ANN_HNSW_INDEX_H_
