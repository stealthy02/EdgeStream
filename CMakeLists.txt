cmake_minimum_required(VERSION 3.16)

project(EdgeStreamRT LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(EDGESTREAM_REQUIRE_RKNN "Fail configuration when librknnrt.so is unavailable" OFF)

# 所有可执行文件统一输出到构建根目录，保持 ./build/<binary> 的扁平调用路径。
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(Threads REQUIRED)
enable_testing()

# ===================== 架构 → ONNX Runtime 目录 =====================
if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ONNX_ARCH_FOLDER "onnxruntime-linux-x64-1.28.0")
elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ONNX_ARCH_FOLDER "onnxruntime-linux-aarch64-1.28.0")
else()
    message(FATAL_ERROR "不支持的架构: ${CMAKE_SYSTEM_PROCESSOR}, 仅支持 x86_64 / aarch64")
endif()

set(ONNXRUNTIME_ROOT "${PROJECT_SOURCE_DIR}/third_party/${ONNX_ARCH_FOLDER}")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib")

if (NOT EXISTS ${ONNXRUNTIME_ROOT})
    message(FATAL_ERROR "找不到对应架构的ONNX Runtime目录: ${ONNXRUNTIME_ROOT}")
endif()

find_package(OpenCV REQUIRED)

# ===================== 检测 librknnrt.so =====================
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

# ===================== 公共 include（INTERFACE）=====================
# 仅承载头文件搜索路径，供各 OBJECT 模块编译与最终库/可执行文件使用。
add_library(edgestream_deps INTERFACE)
target_include_directories(edgestream_deps INTERFACE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/rknn
    ${ONNXRUNTIME_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
)

# nlohmann_json
find_package(nlohmann_json)
if(nlohmann_json_FOUND)
    message(STATUS "✅ 找到 nlohmann_json")
    if(TARGET nlohmann_json::nlohmann_json)
        target_link_libraries(edgestream_deps INTERFACE nlohmann_json::nlohmann_json)
    elseif(DEFINED nlohmann_json_INCLUDE_DIRS)
        target_include_directories(edgestream_deps INTERFACE ${nlohmann_json_INCLUDE_DIRS})
    endif()
else()
    message(WARNING "⚠️ 未找到 nlohmann_json，将跳过 edgestream_core 及相关应用")
endif()

add_subdirectory(src)
add_subdirectory(test)
