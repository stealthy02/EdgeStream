# ===================== 回归测试 =====================
# thread_safe_queue_test：纯 C++ 单元测试，不依赖 RKNN/ONNX，纳入 ctest。
# 仅需 include 搜索路径（edgestream_deps 承载）+ Threads，刻意不链 edgestream_core
# 以保证在无 nlohmann_json / 无 librknnrt 的环境下也能独立构建。
add_executable(thread_safe_queue_test thread_safe_queue_test.cpp)
target_link_libraries(thread_safe_queue_test PRIVATE edgestream_deps Threads::Threads)
add_test(NAME thread_safe_queue_regression COMMAND thread_safe_queue_test)

# rknn_init_test：依赖 librknnrt.so，仅板端可构建，不进 ctest（需手动准备 .rknn 模型）。
if(HAS_RKNN_RT)
    add_executable(rknn_init_test rknn_init_test.cpp)
    target_include_directories(rknn_init_test PRIVATE ${PROJECT_SOURCE_DIR}/include/rknn)
    target_link_libraries(rknn_init_test PRIVATE ${RKNN_RT_LIB})
endif()
