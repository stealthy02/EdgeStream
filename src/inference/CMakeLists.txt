add_library(es_inference OBJECT
    onnx_engine.cpp
    rknn_engine.cpp
)
target_link_libraries(es_inference PUBLIC edgestream_deps)
