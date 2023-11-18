cmake_minimum_required(VERSION 3.17)
project(image_retrieval)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

enable_testing()
option(ENABLE_AVX "use avx" ON)

if(ENABLE_AVX)
    add_definitions(-D_ENABLE_AVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma")
endif()


include_directories(.)
include_directories(third_party)

find_package(benchmark REQUIRED)
find_package(GTest REQUIRED)

find_package(absl REQUIRED)
set(ABSL_LIBRARY
        absl::algorithm
        absl::base
        absl::container
        absl::debugging
        absl::flat_hash_map
        absl::flags
        absl::hash
        absl::memory
        absl::meta
        absl::numeric
        absl::random_random
        absl::strings
        absl::str_format
        absl::synchronization
        absl::time
        absl::utility
        )

set(Protobuf_USE_STATIC_LIBS ON)
find_package(Protobuf REQUIRED)
include_directories(
        ${CMAKE_CURRENT_BINARY_DIR}
        ${Protobuf_INCLUDE_DIRS}
)
SET(CPP_PROTO_META_BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
SET(PY_PROTO_META_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
LIST(APPEND PROTO_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR})
FILE(GLOB_RECURSE PROTO_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/*.proto"
        )
EXECUTE_PROCESS(
        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} ${PROTO_FLAGS}
        --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
        --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
        --python_out=${PY_PROTO_META_BASE_DIR}
        ${PROTO_FILES}
)

FILE(GLOB_RECURSE PROTO_SRCS
        "${CMAKE_CURRENT_BINARY_DIR}/*.cc"
)
message("--- ${PROTO_SRCS}")

add_subdirectory(image_retrieval/ann)
add_subdirectory(image_retrieval/clustering)
add_subdirectory(image_retrieval/concurrency)
add_subdirectory(image_retrieval/feature_extraction)
