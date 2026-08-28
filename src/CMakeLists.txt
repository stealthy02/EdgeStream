# edgestream_core 包含 core/jsonl.cpp（依赖 nlohmann_json），故整棵子树需要它。
if(NOT nlohmann_json_FOUND)
    message(STATUS "⚠️ 跳过 edgestream_core 及应用（缺少 nlohmann_json）")
    return()
endif()

add_subdirectory(core)
add_subdirectory(inference)
add_subdirectory(yolo)
add_subdirectory(app)

# ===================== 核心静态库 =====================
add_library(edgestream_core STATIC
    $<TARGET_OBJECTS:es_core>
    $<TARGET_OBJECTS:es_inference>
    $<TARGET_OBJECTS:es_yolo>
)
target_link_libraries(edgestream_core PUBLIC edgestream_deps)
target_link_directories(edgestream_core PUBLIC ${ONNXRUNTIME_LIB})
target_link_libraries(edgestream_core
    PUBLIC
        ${OpenCV_LIBS}
        onnxruntime
        Threads::Threads
)
if(HAS_RKNN_RT)
    target_link_libraries(edgestream_core PUBLIC ${RKNN_RT_LIB})
endif()
