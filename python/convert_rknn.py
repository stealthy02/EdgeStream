#!/usr/bin/env python3
"""Convert an ONNX model to RKNN without leaving Toolkit scratch files behind."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/onnx/yolo11s_640_split.onnx"))
    parser.add_argument("--output", type=Path, default=Path("models/rknn/yolo11s_640_split_int8.rknn"))
    parser.add_argument("--dataset", type=Path, default=Path("corrected_data/calib.txt"))
    parser.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-platform", default="rk3588")
    args = parser.parse_args()

    model = args.model.resolve()
    output = args.output.resolve()
    dataset = args.dataset.resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    if args.quantize and not dataset.is_file():
        raise FileNotFoundError(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)

    # RKNN Toolkit writes check*.onnx files into the process working directory.
    # Keep that implementation detail in a temporary directory, then remove it.
    original_cwd = Path.cwd()
    with TemporaryDirectory(prefix="edgestream-rknn-") as work_dir:
        os.chdir(work_dir)
        from rknn.api import RKNN

        rknn = RKNN(verbose=True)
        try:
            ret = rknn.config(
                target_platform=args.target_platform,
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
            )
            if ret != 0:
                return ret
            ret = rknn.load_onnx(model=str(model))
            if ret != 0:
                return ret
            build_args = {"do_quantization": args.quantize}
            if args.quantize:
                build_args["dataset"] = str(dataset)
            ret = rknn.build(**build_args)
            if ret != 0:
                return ret
            ret = rknn.export_rknn(str(output))
            print(f"RKNN exported: {output}" if ret == 0 else "RKNN export failed")
            return ret
        finally:
            rknn.release()
            os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
