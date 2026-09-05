#!/usr/bin/env python3
"""Build the single human-readable performance report from benchmark stats."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BACKENDS = ("onnx", "rknn_fp16", "rknn_int8")
LABELS = {
    "onnx": "ONNX",
    "rknn_fp16": "RKNN FP16",
    "rknn_int8": "RKNN INT8",
}


def number(data: dict[str, Any], *path: str) -> float | None:
    current: Any = data
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return float(current) if isinstance(current, (int, float)) else None


def fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def load_stats(root: Path, backend: str) -> dict[str, Any]:
    path = root / f"{backend}_stats.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing benchmark stats: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("frame_count", 0)) != 300:
        raise ValueError(f"{path}: frame_count must be 300")
    for field in ("mean", "p50", "p95", "p99"):
        if number(data, "frame_total", field) is None:
            raise ValueError(f"{path}: missing frame_total.{field}")
    return data


def report_row(backend: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": backend,
        "label": LABELS.get(backend, backend),
        "frame_count": int(stats["frame_count"]),
        "processing_fps": number(stats, "processing_fps"),
        "frame_total_ms": {
            name: number(stats, "frame_total", name)
            for name in ("mean", "p50", "p95", "p99")
        },
        "infer_mean_ms": number(stats, "infer", "mean"),
        "preprocess_mean_ms": number(stats, "preprocess", "mean"),
        "postprocess_mean_ms": number(stats, "postprocess", "mean"),
        "temperature_c": number(stats, "system", "max_temperature_c", "mean"),
        "npu_frequency_mhz": number(stats, "system", "npu_frequency_mhz", "mean"),
        "run": stats.get("run", {}),
    }


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# EdgeStream Performance Report",
        "",
        "| Backend | Frames | Processing FPS | Mean ms | p50 ms | p95 ms | p99 ms | Infer ms | Prep ms | Post ms | Temp C | NPU MHz |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        total = row["frame_total_ms"]
        lines.append(
            "| {label} | {frames} | {fps} | {mean} | {p50} | {p95} | {p99} | {infer} | {prep} | {post} | {temp} | {npu} |".format(
                label=row["label"],
                frames=row["frame_count"],
                fps=fmt(row["processing_fps"]),
                mean=fmt(total["mean"]),
                p50=fmt(total["p50"]),
                p95=fmt(total["p95"]),
                p99=fmt(total["p99"]),
                infer=fmt(row["infer_mean_ms"]),
                prep=fmt(row["preprocess_mean_ms"]),
                post=fmt(row["postprocess_mean_ms"]),
                temp=fmt(row["temperature_c"], 1),
                npu=fmt(row["npu_frequency_mhz"], 1),
            )
        )
    lines.extend([
        "",
        "`Processing FPS` is derived from processing time after a frame leaves the queue; it excludes video decode and queue wait time.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the benchmark summary report")
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/benchmark"))
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    args = parser.parse_args()

    rows = [report_row(backend, load_stats(args.input_dir, backend)) for backend in args.backends]
    summary = {"unit": "ms", "backends": rows}
    (args.input_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.input_dir / "summary.md").write_text(render_markdown(rows), encoding="utf-8")
    print(f"Performance report: {args.input_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
