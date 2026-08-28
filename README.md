# EdgeStream

EdgeStream 是一个面向 RKNN 设备的 YOLO11 推理工程，包含 C++ 推理运行时、C++ 回归/benchmark 工具，以及 Python 模型导出与转换流水线。

- C++ 推理与评测：OrangePi (aarch64，板载 `librknnrt.so` + `onnxruntime-linux-aarch64`)
- Python 模型导出/参考生成：PC (x86_64，`rknn-toolkit2`)

## 快速开始

- [构建说明](docs/BUILD.md)
- [可重复运行与回归说明](docs/REGRESSION.md)
- [模型、输入和设备清单](artifacts/model_manifest.json)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

依赖与构建目标说明详见 [docs/BUILD.md](docs/BUILD.md)。核心目标：`edgestream_core`（共享静态库）、`accuracy_eval`、`video_benchmark`、`edgestream`（占位）、`thread_safe_queue_test`、`rknn_init_test`。

## 目录结构

```
EdgeStream/
├── CMakeLists.txt                 顶层：project/options/find_package/依赖检测/add_subdirectory
├── include/edgestream/            公共库头（按模块组织）
│   ├── core/{file_io,timing,signal,coco_names,jsonl}.h
│   ├── inference/{onnx_engine,rknn_engine}.h
│   ├── yolo/{preprocess,postprocess}.h
│   └── concurrency/thread_safe_queue.h
├── include/rknn/                  RKNN 厂商头文件
├── src/
│   ├── CMakeLists.txt             聚合 edgestream_core
│   ├── core/{file_io,timing,signal,coco_names,jsonl}.cpp
│   ├── inference/{onnx_engine,rknn_engine}.cpp
│   ├── yolo/{preprocess,postprocess}.cpp
│   └── app/{main,accuracy_eval,video_benchmark}.cpp
├── test/{thread_safe_queue_test,rknn_init_test}.cpp
├── python/                        Python 流水线（export_onnx/convert_rknn/generate_reference/yolo_pipeline/export_and_verify_onnx）
├── scripts/                       accuracy_benchmark.py + release_check.sh
├── docs/                          BUILD.md / REGRESSION.md / notes/（归档的中文规划与备忘）
├── models/ assets/ artifacts/ third_party/   模型 / 输入 / 产物 / 第三方
```

## C++ 回归与基准

固定参数：640×640 letterbox、置信度 `0.25`、IoU `0.70`、COCO 80 类。

### 300 帧视频 benchmark（合并为单一 `video_benchmark`）

```bash
./build/video_benchmark --backend onnx      --model models/onnx/yolo11s_640_split.onnx
./build/video_benchmark --backend rknn-fp16 --model models/rknn/yolo11s_640_split.rknn
./build/video_benchmark --backend rknn-int8 --model models/rknn/yolo11s_640_split_int8.rknn
```

输出写入 `--output-dir`（默认 `artifacts/onnx` 或 `artifacts/rknn`）：`<tag>_raw.jsonl`（逐帧耗时 + 阶段 breakdown + 系统快照）+ `<tag>_stats.json`（均值+标准差），`<tag>` ∈ {`onnx`, `rknn_fp16`, `rknn_int8`}。

### 多图四阶段精度评测

```bash
# 阶段1 (PC)：PyTorch 参考 → pt.jsonl
python3 python/generate_reference.py --images-dir assets/regression/coco8

# 阶段2 (板)：C++ 跑 ONNX / RKNN FP16 / RKNN INT8 → 同 run_dir 下三份 jsonl
./build/accuracy_eval --run-dir artifacts/accuracy_eval/<RUN_ID> --images-dir assets/regression/coco8

# 阶段3 (任意)：纯离线汇总报告
python3 scripts/accuracy_benchmark.py --run-dir artifacts/accuracy_eval/<RUN_ID>
```

完整流程、参数与报告说明见 [docs/REGRESSION.md](docs/REGRESSION.md)。固定图 ONNX 导出验证：

```bash
python3 python/export_and_verify_onnx.py --model models/onnx/yolo11s_640_split.onnx
```

## Python 模型流水线

```bash
python3 python/export_onnx.py
python3 python/generate_reference.py
python3 python/export_and_verify_onnx.py --model models/onnx/yolo11s_640.onnx
python3 python/export_and_verify_onnx.py --model models/onnx/yolo11s_640_split.onnx

# RKNN FP16
python3 python/convert_rknn.py --model models/onnx/yolo11s_640_split.onnx \
  --output models/rknn/yolo11s_640_split.rknn --no-quantize

# RKNN INT8
python3 python/convert_rknn.py --model models/onnx/yolo11s_640_split.onnx \
  --output models/rknn/yolo11s_640_split_int8.rknn \
  --dataset corrected_data/calib.txt
```

`python/yolo_pipeline.py` 为共享库（letterbox / 解码 / NMS / JSONL 读写 / `compare_detections` 容差比对），供 `export_and_verify_onnx.py` 等复用。

## 输入约定

RKNN 输入由 C++ 显式准备，`pass_through=1`：FP16 使用 `float16/NHWC`；INT8 使用 `int8/NHWC`，`scale=0.00392157`、`zero_point=-128`；ONNX Runtime 使用 `float32/NCHW`。输入值域均为 `[0,1]`（INT8 再按量化参数转换）。完整模型文件、输出形状和设备信息见 [artifacts/model_manifest.json](artifacts/model_manifest.json)。

双输出 ONNX/RKNN 模型为 `[1,4,8400] + [1,80,8400]`，仅拆分 box 与 class score 的量化范围，不改变 YOLO 语义。

## 已保存结果

- `artifacts/reference/`：PyTorch 参考结果（[fixed_image_comparison.json](artifacts/reference/fixed_image_comparison.json)）
- `artifacts/onnx/` / `artifacts/rknn/`：固定图输出与 300 帧摘要
- `artifacts/benchmark/`：基准清单与运行元数据（[orangepi_300f_manifest.json](artifacts/benchmark/orangepi_300f_manifest.json)、[performance_analysis.md](artifacts/benchmark/performance_analysis.md)）
- `artifacts/accuracy_eval/`：四阶段精度流水线 run 目录
