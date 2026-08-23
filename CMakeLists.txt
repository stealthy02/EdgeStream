cmake_minimum_required(VERSION 3.16)

project(EdgeStreamRT LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
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

# 校验文件夹是否存在，防止漏放库文件
if (NOT EXISTS ${ONNXRUNTIME_ROOT})
    message(FATAL_ERROR "找不到对应架构的ONNX Runtime目录: ${ONNXRUNTIME_ROOT}")
endif()
find_package(OpenCV REQUIRED)
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
        ${PROJECT_SOURCE_DIR}/include/rknn
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(edgestream
    src/main.cpp
)


target_link_directories(edgestream_core PUBLIC
    ${ONNXRUNTIME_LIB}
)

target_link_libraries(edgestream_core
    PUBLIC
        ${OpenCV_LIBS}
        onnxruntime
        /usr/lib/librknnrt.so
)

target_link_libraries(edgestream PRIVATE
    edgestream_core
)


find_package(nlohmann_json REQUIRED)
# 旧版ubuntu包没有命名空间target，直接用头文件路径即可
target_include_directories(edgestream PRIVATE ${nlohmann_json_INCLUDE_DIRS})



add_executable(
    bus_test
    "test/bus_test.cpp"
)

target_link_libraries(bus_test PRIVATE ${nlohmann_json_INCLUDE_DIRS} edgestream_core)

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
