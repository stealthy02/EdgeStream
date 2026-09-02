#!/usr/bin/env python3
"""
【阶段 1 / 3】PyTorch (Ultralytics) 批量参考推理。

作用：用 PyTorch 原生 YOLO 对一批图片做推理，输出每张图的 pytorch/{stem}.jsonl 参考结果，
     作为后续 C++ (ONNX / RKNN FP16 / RKNN INT8) 结果的对比基准。

输出目录结构（与 accuracy_eval C++ 多图版、accuracy_benchmark.py 离线脚本 三方对齐：
按后端分文件夹，固定无日期，可单独重跑而不影响其他后端）：
  artifacts/accuracy_eval/
    ├── pytorch/                 ← 本脚本写
    │   ├── {image_stem_1}.jsonl
    │   ├── {image_stem_2}.jsonl
    │   └── manifest.json        (运行参数、环境信息、图片列表)
    ├── onnx/                    ← C++ 阶段 2 写
    ├── rknn_fp16/               ← C++ 阶段 2 写
    └── rknn_int8/               ← C++ 阶段 2 写

用法示例：
  python3 python/generate_reference.py --images-dir assets/regression/test_image

  # 兼容旧用法：只跑单张 bus.jpg
  python3 python/generate_reference.py --image assets/regression/bus.jpg
"""
from __future__ import annotations
import argparse, json, platform
from datetime import datetime
from pathlib import Path
from typing import cast
import cv2
import numpy as np
import torch, ultralytics
from ultralytics import YOLO
from ultralytics.engine.results import Results
from yolo_pipeline import letterbox, save_jsonl

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def detect_one(model: YOLO, image: np.ndarray, imgsz: int,
               conf: float, iou: float, device: str
               ) -> tuple[list[dict], Results]:
    """跑单张图，返回 (detections_list, ultralytics Results)，detections 的 xyxy 已还原到原图坐标。"""
    boxed, scale, left, top = letterbox(image, imgsz)
    raw_results = list(model(boxed, conf=conf, iou=iou, device=device, verbose=False))
    if not raw_results:
        raise RuntimeError("模型推理返回空结果")
    result = cast(Results, raw_results[0])
    detections: list[dict] = []
    boxes = result.boxes
    if boxes is not None:
        for index in range(len(boxes)):
            xyxy = boxes.xyxy[index].detach().cpu().numpy().astype(float)
            # letterbox 反变换：(coord - pad) / scale，钳位到原图范围
            xyxy[[0, 2]] = np.clip((xyxy[[0, 2]] - left) / scale, 0, image.shape[1])
            xyxy[[1, 3]] = np.clip((xyxy[[1, 3]] - top) / scale, 0, image.shape[0])
            class_id = int(boxes.cls[index].item())
            detections.append({
                "detection_index": index,
                "class_id": class_id,
                "class_name": str(result.names[class_id]),
                "confidence": float(boxes.conf[index].item()),
                "xyxy": [float(v) for v in xyxy.tolist()],
            })
    return detections, result


def main() -> int:
    p = argparse.ArgumentParser(description="YOLO PT 批量参考推理 (阶段 1)")
    p.add_argument("--model", type=Path, default=Path("models/pytorch/yolo11s.pt"))
    # ---- 两种输入模式：单图 / 目录 ----
    p.add_argument("--image", type=Path, default=None,
                   help="单图模式：指定一张图片（兼容旧用法）")
    p.add_argument("--images-dir", type=Path, default=None,
                   help=f"目录模式：扫描目录下所有 {','.join(sorted(IMG_EXTS))} 图片（推荐）")
    # ---- 输出 ----
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/accuracy_eval/pytorch"),
                   help="PT 参考输出目录 (默认 artifacts/accuracy_eval/pytorch，固定无日期，覆盖式写)")
    # ---- 推理超参 ----
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--device", default="cpu", help="cpu / cuda / 0")
    a = p.parse_args()

    if not a.model.is_file():
        raise FileNotFoundError(f"PT 模型不存在: {a.model}")

    # 1) 收集待推理图片
    images: list[Path] = []
    if a.images_dir is not None:
        if not a.images_dir.is_dir():
            raise FileNotFoundError(f"图片目录不存在: {a.images_dir}")
        images = sorted([p for p in a.images_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in IMG_EXTS])
        if not images:
            raise RuntimeError(f"目录 {a.images_dir} 下没有找到可用图片")
    elif a.image is not None:
        if not a.image.is_file():
            raise FileNotFoundError(f"图片不存在: {a.image}")
        images = [a.image]
    else:
        default_dir = Path("assets/regression/test_image")
        if default_dir.is_dir():
            print(f"[信息] 未指定 --image/--images-dir，使用默认目录: {default_dir}")
            images = sorted([p for p in default_dir.iterdir()
                             if p.is_file() and p.suffix.lower() in IMG_EXTS])
        else:
            # 再退化到旧默认 bus.jpg
            default_bus = Path("assets/regression/bus.jpg")
            if default_bus.is_file():
                print(f"[信息] 未指定输入，回退到旧默认单图: {default_bus}")
                images = [default_bus]
            else:
                raise RuntimeError("请通过 --image 或 --images-dir 指定输入图片")
    print(f"[阶段1-PT] 共 {len(images)} 张待推理")

    # 2) 准备输出目录（固定无日期，覆盖式写）
    a.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[阶段1-PT] 输出目录: {a.output_dir}")

    # 3) 加载模型（只加载一次，重复用）
    print(f"[阶段1-PT] 加载模型: {a.model}  (device={a.device}) ...")
    model = YOLO(str(a.model))

    # 4) 逐图推理
    per_image_meta: dict[str, dict] = {}
    for idx, img_path in enumerate(images, 1):
        stem = img_path.stem
        print(f"  [{idx}/{len(images)}] {img_path.name} ... ", end="", flush=True)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[跳过] 无法读取")
            continue
        detections, result = detect_one(model, img, a.imgsz, a.conf, a.iou, a.device)
        save_jsonl(a.output_dir / f"{stem}.jsonl", detections)
        per_image_meta[stem] = {
            "image_path": str(img_path.resolve()),
            "image_size": {"width": int(img.shape[1]), "height": int(img.shape[0])},
            "detection_count": len(detections),
        }
        print(f"OK  ({len(detections)} dets)")

    # 5) manifest
    manifest = {
        "stage": "pt_reference",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "env": {
            "python": platform.python_version(),
            "system": f"{platform.system()} {platform.release()}",
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "args": {
            "model": str(a.model),
            "imgsz": a.imgsz,
            "conf": a.conf,
            "iou": a.iou,
            "device": a.device,
        },
        "images": [p.stem for p in images],
        "per_image": per_image_meta,
    }
    (a.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[阶段1-PT] 完成。manifest: {a.output_dir / 'manifest.json'}")
    print(f"[提示] 接下来跑阶段 2 (C++) 和阶段 3，默认都对齐 artifacts/accuracy_eval：")
    print(f"  ./accuracy_eval --images-dir {a.images_dir or a.image}   # → onnx/ rknn_fp16/ rknn_int8/")
    print(f"  python3 scripts/accuracy_benchmark.py                   # → summary.json/.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
