# edgestream：main.cpp 占位程序（逻辑待后续完善），仅依赖 RKNN 运行时。
if(HAS_RKNN_RT)
    add_executable(edgestream main.cpp)
    target_link_libraries(edgestream PRIVATE edgestream_core)
endif()

# 阶段 2 多图精度评测 + 视频 benchmark：需要 RKNN 运行时与 nlohmann_json。
if(HAS_RKNN_RT AND nlohmann_json_FOUND)
    add_executable(accuracy_eval accuracy_eval.cpp)
    target_link_libraries(accuracy_eval PRIVATE edgestream_core)

    add_executable(video_benchmark video_benchmark.cpp)
    target_link_libraries(video_benchmark PRIVATE edgestream_core)

    # 原始张量导出（无后处理），配合 scripts/raw_eval_check.py 做 MAPE 校验。
    add_executable(raw_eval raw_eval.cpp)
    target_link_libraries(raw_eval PRIVATE edgestream_core)
endif()
