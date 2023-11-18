
add_library(kmeans kmeans.cc ${PROTO_SRCS})
target_link_libraries(kmeans
        pthread
        thread_pool
        absl::str_format
        absl::time
        ${Protobuf_LIBRARIES}
        )

add_executable(kmeans_test kmeans_test.cc)
target_link_libraries(kmeans_test kmeans
        gtest gtest_main
        )
add_test(clustering_test kmeans_test)

add_executable(clustering clustering.cc ${PROTO_SRCS})
target_link_libraries(clustering kmeans ${Protobuf_LIBRARIES})
