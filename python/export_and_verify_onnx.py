#!/usr/bin/env python3
"""Run the ONNX model (non-split) in pure Python and compare decoded detections
with the PyTorch reference, to isolate "ONNX 导出本身坏没坏" 与 C++ 代码无关。

参考结果必须先由 generate_reference.py 生成（与图片同名）：
  python3 python/generate_reference.py --image assets/regression/bus.jpg
  → artifacts/accuracy_eval/pytorch/bus.jsonl
然后本脚本读它做对比。
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from yolo_pipeline import (build_input_tensor, compare_detections, decode_outputs,
                           letterbox, load_jsonl, save_jsonl)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=Path('models/onnx/yolo11s_640.onnx'))
    p.add_argument('--image', type=Path, default=Path('assets/regression/bus.jpg'))
    p.add_argument('--reference', type=Path, default=None,
                   help='PyTorch 参考 JSONL (默认 artifacts/accuracy_eval/pytorch/{image_stem}.jsonl)')
    p.add_argument('--output-dir', type=Path, default=Path('artifacts/onnx_verify'))
    p.add_argument('--imgsz', type=int, default=640); p.add_argument('--conf', type=float, default=.25); p.add_argument('--iou', type=float, default=.70)
    p.add_argument('--score-atol', type=float, default=1e-3); p.add_argument('--box-atol', type=float, default=2.)
    a = p.parse_args()
    # 参考路径默认按图片名推导到新结构 pytorch/{stem}.jsonl
    if a.reference is None:
        a.reference = Path('artifacts/accuracy_eval/pytorch') / f'{a.image.stem}.jsonl'
    # 分开报错，缺参考时直接告诉用户怎么补
    if not a.model.is_file():     raise FileNotFoundError(f'模型不存在: {a.model}')
    if not a.image.is_file():     raise FileNotFoundError(f'图片不存在: {a.image}')
    if not a.reference.is_file():
        raise FileNotFoundError(
            f'参考结果不存在: {a.reference}\n'
            f'  请先跑阶段 1 生成 PyTorch 参考：\n'
            f'  python3 python/generate_reference.py --image {a.image}'
        )
    loaded_image = cv2.imread(str(a.image))
    if loaded_image is None:
        raise RuntimeError(f'Failed to read image: {a.image}')
    image = np.asarray(loaded_image)
    boxed, scale, left, top = letterbox(image, a.imgsz)
    session = ort.InferenceSession(str(a.model), providers=['CPUExecutionProvider'])
    raw_outputs = session.run(None, {session.get_inputs()[0].name: build_input_tensor(boxed)})
    outputs = [np.asarray(output) for output in raw_outputs]
    boxes, scores, classes = decode_outputs(outputs, a.conf, a.iou, scale, left, top, image.shape[1], image.shape[0])
    reference = load_jsonl(a.reference); names = {int(x['class_id']): x.get('class_name', f"class_{x['class_id']}") for x in reference}
    actual = [{'detection_index': i, 'class_id': int(c), 'class_name': names.get(int(c), f'class_{c}'), 'confidence': float(s), 'xyxy': [float(v) for v in b]} for i, (b, s, c) in enumerate(zip(boxes, scores, classes))]
    a.output_dir.mkdir(parents=True, exist_ok=True); save_jsonl(a.output_dir / f'{a.image.stem}.jsonl', actual)
    result = compare_detections(reference, actual, a.score_atol, a.box_atol)
    (a.output_dir / f'{a.image.stem}_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'model: {a.model}\noutputs: {len(outputs)}\ndetections: {len(actual)}\npassed: {result["passed"]}')
    return 0 if result['passed'] else 1

if __name__ == '__main__': raise SystemExit(main())
