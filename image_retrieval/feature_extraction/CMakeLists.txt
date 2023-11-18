
add_executable(feature_decoder feature_decoder.cc ${PROTO_SRCS})
target_link_libraries(feature_decoder ${Protobuf_LIBRARIES})
