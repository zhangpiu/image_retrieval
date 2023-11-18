
add_library(thread_pool thread_pool.cc)
target_link_libraries(thread_pool
        absl::str_format
        absl::synchronization
        absl::time
        )
