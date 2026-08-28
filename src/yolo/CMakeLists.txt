add_library(es_yolo OBJECT
    preprocess.cpp
    postprocess.cpp
)
target_link_libraries(es_yolo PUBLIC edgestream_deps)
