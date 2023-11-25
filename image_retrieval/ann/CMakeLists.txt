
add_library(binary_index binary_index.cc ${PROTO_SRCS})
target_link_libraries(binary_index
        thread_pool
        absl::str_format
        absl::time
        ${Protobuf_LIBRARIES}
        )

add_library(flat_index flat_index.cc ${PROTO_SRCS})
target_link_libraries(flat_index
        thread_pool
        absl::str_format
        absl::time
        ${Protobuf_LIBRARIES}
        )

add_library(hnsw_index hnsw_index.cc ${PROTO_SRCS})
target_link_libraries(hnsw_index
        thread_pool
        absl::str_format
        absl::time
        ${Protobuf_LIBRARIES}
)

add_executable(search_engine search_engine.cc)
target_link_libraries(search_engine flat_index binary_index hnsw_index)

add_executable(vector_distance_test vector_distance_test.cc)
target_link_libraries(vector_distance_test
        absl::random_random
        gtest gtest_main
        )
add_test(ann_test vector_distance_test)

add_executable(vector_distance_benchmark vector_distance_benchmark.cc)
target_link_libraries(vector_distance_benchmark
        absl::random_random
        benchmark
        )
