#!/usr/bin/env bash
# Run the complete board regression. It only updates the two canonical reports:
# artifacts/accuracy_eval/summary.md and artifacts/benchmark/summary.md.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${PROJECT_ROOT}/build"
ACCURACY_DIR="${PROJECT_ROOT}/artifacts/accuracy_eval"
BENCHMARK_DIR="${PROJECT_ROOT}/artifacts/benchmark"

run() {
    echo
    echo "+ $*"
    "$@"
}

require_file() {
    [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

require_file "${PROJECT_ROOT}/models/onnx/yolo11s_640_split.onnx"
require_file "${PROJECT_ROOT}/models/rknn/yolo11s_640_split.rknn"
require_file "${PROJECT_ROOT}/models/rknn/yolo11s_640_split_int8.rknn"
require_file "${PROJECT_ROOT}/assets/regression/input.mp4"
[[ -d "${PROJECT_ROOT}/assets/regression/test_image" ]] || {
    echo "Missing regression image directory" >&2
    exit 1
}
[[ -d "${ACCURACY_DIR}/pytorch" ]] || {
    echo "Missing PyTorch reference results: ${ACCURACY_DIR}/pytorch" >&2
    echo "Run python3 python/generate_reference.py --images-dir assets/regression/test_image on the PC first." >&2
    exit 1
}

run cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release -DEDGESTREAM_REQUIRE_RKNN=ON
run cmake --build "${BUILD_DIR}" -j"$(nproc)"
run ctest --test-dir "${BUILD_DIR}" --output-on-failure

# Keep the PyTorch reference, but discard C++ outputs so a failed run can never
# be mistaken for a previous successful result.
rm -rf "${ACCURACY_DIR}/onnx" "${ACCURACY_DIR}/rknn_fp16" "${ACCURACY_DIR}/rknn_int8"
rm -f "${ACCURACY_DIR}/summary.json" "${ACCURACY_DIR}/summary.md"

run "${BUILD_DIR}/accuracy_eval" \
    --images-dir "${PROJECT_ROOT}/assets/regression/test_image" \
    --output-root "${ACCURACY_DIR}"
run python3 "${PROJECT_ROOT}/scripts/accuracy_benchmark.py" \
    --output-root "${ACCURACY_DIR}" \
    --require-stages onnx rknn_fp16 rknn_int8

mkdir -p "${BENCHMARK_DIR}"
run "${BUILD_DIR}/video_benchmark" --backend onnx \
    --model "${PROJECT_ROOT}/models/onnx/yolo11s_640_split.onnx" \
    --video "${PROJECT_ROOT}/assets/regression/input.mp4" \
    --output-dir "${BENCHMARK_DIR}"
run "${BUILD_DIR}/video_benchmark" --backend rknn-fp16 \
    --model "${PROJECT_ROOT}/models/rknn/yolo11s_640_split.rknn" \
    --video "${PROJECT_ROOT}/assets/regression/input.mp4" \
    --output-dir "${BENCHMARK_DIR}"
run "${BUILD_DIR}/video_benchmark" --backend rknn-int8 \
    --model "${PROJECT_ROOT}/models/rknn/yolo11s_640_split_int8.rknn" \
    --video "${PROJECT_ROOT}/assets/regression/input.mp4" \
    --output-dir "${BENCHMARK_DIR}"
run python3 "${PROJECT_ROOT}/scripts/performance_report.py" \
    --input-dir "${BENCHMARK_DIR}"

echo
echo "Accuracy report:    ${ACCURACY_DIR}/summary.md"
echo "Performance report: ${BENCHMARK_DIR}/summary.md"
