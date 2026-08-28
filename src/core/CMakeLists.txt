add_library(es_core OBJECT
    file_io.cpp
    timing.cpp
    signal.cpp
    coco_names.cpp
    jsonl.cpp
)
target_link_libraries(es_core PUBLIC edgestream_deps)
