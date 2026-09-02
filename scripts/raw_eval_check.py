#!/usr/bin/env python3
"""
【原始输出校验】读取 raw_eval C++ 导出的原始张量（.bin），只算数值误差指标。

对比方式：选定一个后端作为基准（默认 onnx），其余后端逐图与其做逐元素对比。
只算 MAPE 之类的张量级指标，不做解码 / NMS / 框匹配：
    MAPE(%)   平均绝对百分比误差，分母用 max(|ref|, EPS) 防除零
    CosSim    展平后的余弦相似度
    MaxAbs    最大绝对误差
    MeanAbs   平均绝对误差

输入目录（与 raw_eval C++ 约定对齐，固定无日期，覆盖式写）：
  artifacts/raw_eval/
    ├── onnx/      {stem}.bin + manifest.json
    ├── rknn_fp16/ {stem}.bin + manifest.json
    └── rknn_int8/ {stem}.bin + manifest.json

用法示例：
  # 默认：onnx 为基准，对比其余所有后端，全部图片
  python3 scripts/raw_eval_check.py

  # 只对比指定后端 / 限定图片数量：
  python3 scripts/raw_eval_check.py --backends rknn_int8 --num 16

  # 换基准 / 看逐图明细：
  python3 scripts/raw_eval_check.py --ref rknn_fp16 --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("artifacts/raw_eval")
DEFAULT_BACKENDS = ["onnx", "rknn_fp16", "rknn_int8"]
EPS = 1e-6


def load_backend(root: Path, backend: str) -> dict[str, np.ndarray]:
    """读取一个后端目录的全部 .bin，返回 {stem: ndarray(float32, 按 manifest shape reshape)}"""
    d = root / backend
    if not d.is_dir():
        raise FileNotFoundError(f"后端目录不存在: {d}")

    manifest_path = d / "manifest.json"
    tensors: dict[str, np.ndarray] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for stem, meta in manifest.get("outputs", {}).items():
            arr = np.fromfile(d / meta["file"], dtype=np.float32)
            shape = meta.get("shape") or [arr.size]
            if int(np.prod(shape)) == arr.size:
                arr = arr.reshape(shape)
            tensors[stem] = arr
    else:
        # 没有 manifest 就按平铺一维读
        for f in sorted(d.glob("*.bin")):
            tensors[f.stem] = np.fromfile(f, dtype=np.float32)
    return tensors


def compute_metrics(ref: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """逐元素误差指标（输入为一维 float32 数组，长度一致）。"""
    diff = np.abs(ref.astype(np.float64) - x.astype(np.float64))
    abs_ref = np.abs(ref.astype(np.float64))
    mape = float(np.mean(diff / np.maximum(abs_ref, EPS)) * 100.0)
    denom = float(np.linalg.norm(ref) * np.linalg.norm(x)) + EPS
    cos = float(np.dot(ref, x) / denom)
    return {
        "mape": mape,
        "cos": cos,
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="raw_eval 原始张量校验（只算 MAPE 等数值指标）")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help=f"raw_eval 输出根目录 (默认 {DEFAULT_ROOT})")
    p.add_argument("--ref", default="onnx",
                   help="对比基准后端 (默认 onnx)")
    p.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS,
                   help="待校验的后端列表 (默认 onnx rknn_fp16 rknn_int8)")
    p.add_argument("--num", type=int, default=None,
                   help="限制参与对比的图片数量 (默认全部)")
    p.add_argument("--verbose", action="store_true",
                   help="额外打印逐图指标明细")
    a = p.parse_args()

    if not a.root.is_dir():
        print(f"❌ 输出根目录不存在: {a.root}（先跑 ./raw_eval）")
        return 1

    # 加载基准 + 各后端
    try:
        ref_tensors = load_backend(a.root, a.ref)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    if not ref_tensors:
        print(f"❌ 基准后端 {a.ref} 目录里没有 .bin 数据")
        return 1

    others: dict[str, dict[str, np.ndarray]] = {}
    for backend in a.backends:
        if backend == a.ref:
            continue
        try:
            tensors = load_backend(a.root, backend)
        except FileNotFoundError as e:
            print(f"⚠️  跳过 {e}")
            continue
        if tensors:
            others[backend] = tensors

    if not others:
        print("⚠️  没有可对比的后端（至少需要基准 + 1 个其他后端）")
        return 1

    print("=" * 64)
    print("【原始输出校验】raw_eval_check")
    print(f"  root = {a.root}")
    print(f"  ref  = {a.ref} ({len(ref_tensors)} 个张量)")
    print("=" * 64)

    # 逐后端 × 逐图对比
    all_rows: dict[str, list[dict]] = {}
    for backend, tensors in others.items():
        common = sorted(set(ref_tensors) & set(tensors))
        if a.num is not None:
            common = common[: a.num]
        if not common:
            print(f"⚠️  {backend}: 与基准无共同图片，跳过")
            continue
        rows = []
        for stem in common:
            ref, x = ref_tensors[stem].ravel(), tensors[stem].ravel()
            if ref.size != x.size:
                print(f"⚠️  {backend}/{stem}: 元素数不一致 ({ref.size} vs {x.size})，跳过")
                continue
            rows.append({"stem": stem, **compute_metrics(ref, x)})
        all_rows[backend] = rows

    # 汇总表
    print(f"\n对比基准: {a.ref}\n" + "-" * 64)
    header = f"{'backend':<12}{'images':>7}{'MAPE(%)':>12}{'CosSim':>10}{'MaxAbs':>11}{'MeanAbs':>11}"
    print(header)
    print("-" * 64)
    for backend, rows in all_rows.items():
        agg = {
            "mape": float(np.mean([r["mape"] for r in rows])),
            "cos": float(np.mean([r["cos"] for r in rows])),
            "max_abs": float(np.max([r["max_abs"] for r in rows])),
            "mean_abs": float(np.mean([r["mean_abs"] for r in rows])),
        }
        print(f"{backend:<12}{len(rows):>7}"
              f"{agg['mape']:>12.4f}{agg['cos']:>10.6f}"
              f"{agg['max_abs']:>11.4f}{agg['mean_abs']:>11.6f}")
        if a.verbose:
            for r in rows:
                print(f"  {r['stem']:<24}"
                      f"MAPE={r['mape']:>10.4f}%  Cos={r['cos']:.6f}  "
                      f"MaxAbs={r['max_abs']:.4f}  MeanAbs={r['mean_abs']:.6f}")
    print("-" * 64)
    print("说明: MAPE 分母为 max(|ref|, 1e-6)，接近 0 的元素会放大误差，"
          "请结合 CosSim / MaxAbs 一起看。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
