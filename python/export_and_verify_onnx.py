#!/usr/bin/env python3
"""Run either ONNX variant and compare decoded detections with the reference."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from yolo_pipeline import build_input_tensor, decode_outputs, letterbox, load_jsonl, save_jsonl

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=Path('models/onnx/yolo11s_640.onnx'))
    p.add_argument('--image', type=Path, default=Path('assets/regression/bus.jpg'))
    p.add_argument('--reference', type=Path, default=Path('artifacts/reference/bus_detections.jsonl'))
    p.add_argument('--output-dir', type=Path, default=Path('artifacts/onnx'))
    p.add_argument('--imgsz', type=int, default=640); p.add_argument('--conf', type=float, default=.25); p.add_argument('--iou', type=float, default=.70)
    p.add_argument('--score-atol', type=float, default=1e-3); p.add_argument('--box-atol', type=float, default=2.)
    a = p.parse_args()
    for path in (a.model, a.image, a.reference):
        if not path.is_file(): raise FileNotFoundError(path)
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
    a.output_dir.mkdir(parents=True, exist_ok=True); save_jsonl(a.output_dir / 'onnx_detections.jsonl', actual)
    result = compare(reference, actual, a.score_atol, a.box_atol)
    (a.output_dir / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'model: {a.model}\noutputs: {len(outputs)}\ndetections: {len(actual)}\npassed: {result["passed"]}')
    return 0 if result['passed'] else 1

def compare(reference, actual, score_atol, box_atol):
    unmatched, matches, passed = set(range(len(reference))), [], len(reference) == len(actual)
    for ai, item in enumerate(actual):
        candidates = [i for i in unmatched if int(reference[i]['class_id']) == item['class_id']]
        if not candidates: passed = False; matches.append({'actual_index': ai, 'matched': False}); continue
        ri = min(candidates, key=lambda i: sum((item['xyxy'][j] - reference[i]['xyxy'][j]) ** 2 for j in range(4)))
        unmatched.remove(ri); score_error = abs(item['confidence'] - float(reference[ri]['confidence']))
        box_error = max(abs(item['xyxy'][j] - float(reference[ri]['xyxy'][j])) for j in range(4)); ok = score_error <= score_atol and box_error <= box_atol
        passed &= ok; matches.append({'actual_index': ai, 'reference_index': ri, 'score_abs_error': score_error, 'box_max_abs_error': box_error, 'passed': ok})
    passed &= not unmatched
    return {'passed': passed, 'reference_count': len(reference), 'actual_count': len(actual), 'unmatched_reference_indices': sorted(unmatched), 'reference_class_counts': dict(Counter(x['class_name'] for x in reference)), 'actual_class_counts': dict(Counter(x['class_name'] for x in actual)), 'matches': matches}

if __name__ == '__main__': raise SystemExit(main())
