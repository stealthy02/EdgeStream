
# EdgeStream

EdgeStream 是一个面向 RKNN 设备的 YOLO11 推理工程，包含 C++ 推理运行时、C++ 回归/benchmark 工具，以及 Python 模型导出和转换流水线。

## 快速开始

- [构建说明](docs/BUILD.md)
- [可重复运行与回归说明](docs/REGRESSION.md)
- [模型、输入和设备清单](artifacts/model_manifest.json)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DEDGESTREAM_REQUIRE_RKNN=ON
cmake --build build -j"$(nproc)"
```

## C++ 运行与回归

固定图片为 `assets/regression/bus.jpg`，使用 640x640 letterbox、置信度 `0.25`、IoU `0.70`：

```bash
./build/bus_test --onnx
./build/bus_test --rknn
```

300 帧视频 benchmark 使用 `assets/regression/input.mp4`：

```bash
./build/onnx_video_test models/onnx/yolo11s_640_split.onnx
./build/rknn_video_test models/rknn/yolo11s_640_split.rknn
./build/rknn_video_test models/rknn/yolo11s_640_split_int8.rknn
```

结果保存到 `artifacts/onnx/` 或 `artifacts/rknn/`，包括摘要和逐帧 JSONL。固定图结果对照见 [fixed_image_comparison.json](artifacts/reference/fixed_image_comparison.json)，benchmark 清单见 [orangepi_300f_manifest.json](artifacts/benchmark/orangepi_300f_manifest.json)。

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

## 输入约定

RKNN 输入由 C++ 显式准备，`pass_through=1`：FP16 使用 `float16/NHWC`；INT8 使用 `int8/NHWC`，`scale=0.00392157`、`zero_point=-128`；ONNX Runtime 使用 `float32/NCHW`。输入值域均为 `[0,1]`（INT8 再按量化参数转换）。完整模型文件、输出形状和设备信息见 [model_manifest.json](artifacts/model_manifest.json)。

双输出 ONNX/RKNN 模型为 `[1,4,8400] + [1,80,8400]`，仅拆分 box 与 class score 的量化范围，不改变 YOLO 语义。

## 已保存结果

`artifacts/reference/` 保存 PyTorch 参考结果；`artifacts/onnx/` 和 `artifacts/rknn/` 保存固定图输出及 300 帧摘要；`artifacts/benchmark/` 保存 benchmark 格式和运行元数据。
