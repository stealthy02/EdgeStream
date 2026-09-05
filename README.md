# EdgeStream

EdgeStream 是运行在 RK3588 设备上的 C++17 YOLO11 推理工程。它使用同一个拆分输出模型运行 ONNX Runtime、RKNN FP16 和 RKNN INT8 三个后端，最终只产出两份报告：

- `artifacts/accuracy_eval/summary.md`：PyTorch、ONNX、RKNN FP16、RKNN INT8 的检测结果一致性。
- `artifacts/benchmark/summary.md`：三个 C++ 后端的 300 帧处理时延、p50/p95/p99、FPS 和系统快照。

流程分为两台机器：

- PC：导出模型并生成 PyTorch 参考结果。
- Orange Pi：构建 C++、运行回归并生成两份报告。

下文所有命令均在仓库根目录执行。

## 1. PC：准备模型与参考结果

推荐 Python 3.10，并使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-pc.txt
```

安装 Rockchip 或 Orange Pi 提供的 x86_64 `rknn-toolkit2` wheel，并验证安装：

```bash
python3 -m pip install /path/to/rknn_toolkit2-<version>-cp310-cp310-linux_x86_64.whl
python3 -c "from rknn.api import RKNN; print('rknn-toolkit2 OK')"
```

导出 ONNX，再生成 FP16 与 INT8 RKNN 模型：

```bash
python3 python/export_onnx.py

python3 python/convert_rknn.py \
  --model models/onnx/yolo11s_640_split.onnx \
  --output models/rknn/yolo11s_640_split.rknn \
  --no-quantize

python3 python/convert_rknn.py \
  --model models/onnx/yolo11s_640_split.onnx \
  --output models/rknn/yolo11s_640_split_int8.rknn \
  --dataset corrected_data/calib.txt
```

为回归图像集生成固定的 PyTorch 参考结果：

```bash
python3 python/generate_reference.py \
  --images-dir assets/regression/test_image
```

进入板端前，PC 上应存在：

```text
models/onnx/yolo11s_640_split.onnx
models/rknn/yolo11s_640_split.rknn
models/rknn/yolo11s_640_split_int8.rknn
artifacts/accuracy_eval/pytorch/
```

将仓库、生成的模型和 `artifacts/accuracy_eval/pytorch/` 同步到 Orange Pi；不要复制 PC 上的 `build/` 目录。

## 2. Orange Pi：构建并运行

板端需要 CMake、C++17 编译器、OpenCV 开发包、`nlohmann-json3-dev`、`third_party/` 下匹配 aarch64 的 ONNX Runtime，以及 `/usr/lib/librknnrt.so`。

在 Debian/Ubuntu 系统中安装通用构建依赖：

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev nlohmann-json3-dev
```

运行完整板端回归：

```bash
bash scripts/release_check.sh
```

该命令会配置和构建工程、执行 CTest、重新生成 C++ 精度输出、运行三个 300 帧 benchmark，并写入：

```text
artifacts/accuracy_eval/summary.md
artifacts/benchmark/summary.md
```

`release_check.sh` 会保留 PyTorch 参考结果，只清理和重建 C++ 后端输出。因此，后端失败不会被旧的成功结果掩盖。

## 报告含义

`accuracy_eval/summary.md` 比较类别、数量、置信度和检测框 IoU。只有在模型、图像集、预处理参数、置信度阈值和 IoU 阈值保持不变时，前后两次精度报告才可直接比较。

`benchmark/summary.md` 汇总 ONNX、RKNN FP16 和 RKNN INT8。`Processing FPS` 与帧时延从帧离开队列后开始计算，不包含视频解码和队列等待；逐帧 JSONL 与各后端 stats JSON 保留在报告旁边，仅在指标变化时用于排查。

## 目录说明

```text
include/        公共 C++ 头文件
src/            运行时、推理后端、前后处理和应用入口
test/           单元测试
python/         模型导出、转换和 PyTorch 参考生成
scripts/        精度报告、性能报告和板端回归入口
assets/         回归图像与 benchmark 视频
corrected_data/ INT8 校准集
models/         PyTorch 源模型与生成的 ONNX/RKNN 模型
third_party/    按架构提供的 ONNX Runtime
artifacts/      生成的参考结果、原始数据和报告
```

单独执行各步骤或排查问题时，见 [构建说明](docs/BUILD.md) 和 [回归说明](docs/REGRESSION.md)。
