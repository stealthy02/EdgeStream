cmake_minimum_required(VERSION 3.16)

project(EdgeStreamRT LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(EDGESTREAM_REQUIRE_RKNN "Fail configuration when librknnrt.so is unavailable" OFF)

find_package(Threads REQUIRED)
enable_testing()

add_executable(thread_safe_queue_test
    test/thread_safe_queue_test.cpp
)
target_include_directories(thread_safe_queue_test PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(thread_safe_queue_test PRIVATE Threads::Threads)
add_test(NAME thread_safe_queue_regression COMMAND thread_safe_queue_test)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    # WSL / x64 PC
    set(ONNX_ARCH_FOLDER "onnxruntime-linux-x64-1.28.0")
elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    # 香橙派 ARM64
    set(ONNX_ARCH_FOLDER "onnxruntime-linux-aarch64-1.28.0")
else()
    message(FATAL_ERROR "不支持的架构: ${CMAKE_SYSTEM_PROCESSOR}, 仅支持 x86_64 / aarch64")
endif()

# 自动拼接根路径
set(ONNXRUNTIME_ROOT "${PROJECT_SOURCE_DIR}/third_party/${ONNX_ARCH_FOLDER}")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib")

# 校验 ONNX Runtime 文件夹是否存在
if (NOT EXISTS ${ONNXRUNTIME_ROOT})
    message(FATAL_ERROR "找不到对应架构的ONNX Runtime目录: ${ONNXRUNTIME_ROOT}")
endif()

find_package(OpenCV REQUIRED)

# ===================== 新增：检测 librknnrt.so =====================
set(RKNN_RT_LIB "/usr/lib/librknnrt.so")
set(HAS_RKNN_RT OFF)
if(EXISTS ${RKNN_RT_LIB})
    set(HAS_RKNN_RT ON)
    message(STATUS "✅ 检测到 librknnrt.so : ${RKNN_RT_LIB}")
else()
    if(EDGESTREAM_REQUIRE_RKNN)
        message(FATAL_ERROR "找不到 librknnrt.so: ${RKNN_RT_LIB} (EDGESTREAM_REQUIRE_RKNN=ON)")
    endif()
    message(WARNING "⚠️ 未找到 librknnrt.so : ${RKNN_RT_LIB}")
endif()

# 核心库：始终添加目标（按需求保留edgestream_core）
# 注意：源码包含 rknn_engine.cpp 且链接 librknnrt.so
# 若库缺失，edgestream_core 链接会失败（符合要求：不修改源码绕过依赖）
add_library(edgestream_core
    src/inference_engine.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/rknn_engine.cpp
)

target_include_directories(edgestream_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/rknn
        ${ONNXRUNTIME_ROOT}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(edgestream_core PUBLIC
    ${ONNXRUNTIME_LIB}
)

target_link_libraries(edgestream_core
    PUBLIC
        ${OpenCV_LIBS}
        onnxruntime
        ${RKNN_RT_LIB}
)

if(HAS_RKNN_RT)
    add_executable(edgestream
        src/main.cpp
    )
    target_link_libraries(edgestream PRIVATE
        edgestream_core
    )
    message(STATUS "✅ 将构建目标：edgestream_core, edgestream")
else()
    message(WARNING "⚠️ 跳过构建可执行文件 edgestream（缺失 librknnrt.so）")
    message(STATUS "✅ 将构建目标：edgestream_core")
endif()

find_package(nlohmann_json)
if(nlohmann_json_FOUND)
    message(STATUS "✅ 找到 nlohmann_json，将编译 JSON 相关测试程序")
    add_executable(
        bus_test
        "test/bus_test.cpp"
    )
    target_link_libraries(bus_test PRIVATE edgestream_core)
    target_include_directories(bus_test PRIVATE ${nlohmann_json_INCLUDE_DIRS})

    add_executable(rknn_video_test
        "test/rknn_video_test.cpp"
    )
    target_link_libraries(rknn_video_test PRIVATE
        edgestream_core
    )
    target_include_directories(rknn_video_test PRIVATE ${nlohmann_json_INCLUDE_DIRS})

    add_executable(rknn_int8_video_test
        "test/rknn_int8_video_test.cpp"
    )
    target_link_libraries(rknn_int8_video_test PRIVATE
        edgestream_core
    )
    target_include_directories(rknn_int8_video_test PRIVATE ${nlohmann_json_INCLUDE_DIRS})

    add_executable(onnx_video_test
        "test/onnx_video_test.cpp"
    )
    target_link_libraries(onnx_video_test PRIVATE
        edgestream_core
    )
    target_include_directories(onnx_video_test PRIVATE ${nlohmann_json_INCLUDE_DIRS})
else()
    message(WARNING "⚠️ 未找到 nlohmann_json，跳过编译 bus_test / rknn_video_test / rknn_int8_video_test / onnx_video_test")
endif()
