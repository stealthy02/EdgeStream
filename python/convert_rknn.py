#!/usr/bin/env python3
"""Convert either canonical ONNX variant to RKNN, optionally with INT8 calibration."""
from __future__ import annotations
import argparse
from pathlib import Path
from rknn.api import RKNN

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=Path('models/onnx/yolo11s_640_split.onnx'))
    p.add_argument('--output', type=Path, default=Path('models/rknn/yolo11s_640_split_int8.rknn'))
    p.add_argument('--dataset', type=Path, default=Path('corrected_data/calib.txt'))
    p.add_argument('--quantize', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--target-platform', default='rk3588'); a = p.parse_args()
    if not a.model.is_file(): raise FileNotFoundError(a.model)
    if a.quantize and not a.dataset.is_file(): raise FileNotFoundError(a.dataset)
    a.output.parent.mkdir(parents=True, exist_ok=True); rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(target_platform=a.target_platform, mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]])
        if ret != 0: return ret
        ret = rknn.load_onnx(model=str(a.model));
        if ret != 0: return ret
        kwargs = {'do_quantization': a.quantize};
        if a.quantize: kwargs['dataset'] = str(a.dataset)
        ret = rknn.build(**kwargs);
        if ret != 0: return ret
        ret = rknn.export_rknn(str(a.output)); print(f'RKNN exported: {a.output}' if ret == 0 else 'RKNN export failed'); return ret
    finally: rknn.release()

if __name__ == '__main__': raise SystemExit(main())
