#!/usr/bin/env python3
"""Generate the CPU/Ultralytics reference using the export preprocessing."""
from __future__ import annotations
import argparse, json, platform
from pathlib import Path
from typing import cast
import cv2
import torch, ultralytics
from ultralytics import YOLO
from ultralytics.engine.results import Results
from yolo_pipeline import letterbox

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=Path('models/pytorch/yolo11s.pt'))
    p.add_argument('--image', type=Path, default=Path('assets/regression/bus.jpg'))
    p.add_argument('--output-dir', type=Path, default=Path('artifacts/reference'))
    p.add_argument('--imgsz', type=int, default=640); p.add_argument('--conf', type=float, default=.25)
    p.add_argument('--iou', type=float, default=.70); p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if not a.model.is_file() or not a.image.is_file(): raise FileNotFoundError('model and image must exist')
    image = cv2.imread(str(a.image))
    if image is None:
        raise ValueError(f'failed to read image: {a.image}')
    boxed, scale, left, top = letterbox(image, a.imgsz)
    raw_results = list(YOLO(str(a.model))(boxed, conf=a.conf, iou=a.iou, device=a.device, verbose=False))
    if not raw_results:
        raise RuntimeError('the model returned no results')
    result = cast(Results, raw_results[0])
    detections = []
    boxes = result.boxes
    if boxes is not None:
        for index in range(len(boxes)):
            xyxy = boxes.xyxy[index].detach().cpu().numpy().astype(float)
            xyxy[[0, 2]] = np_clip(xyxy[[0, 2]], left, scale, image.shape[1]); xyxy[[1, 3]] = np_clip(xyxy[[1, 3]], top, scale, image.shape[0])
            class_id = int(boxes.cls[index].item())
            detections.append({'detection_index': index, 'class_id': class_id, 'class_name': str(result.names[class_id]), 'confidence': float(boxes.conf[index].item()), 'xyxy': xyxy.tolist()})
    a.output_dir.mkdir(parents=True, exist_ok=True); path = a.output_dir / 'bus_detections.jsonl'
    with path.open('w', encoding='utf-8') as f:
        for item in detections: f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n')
    metadata = {'model': str(a.model), 'image': str(a.image), 'imgsz': a.imgsz, 'conf': a.conf, 'iou': a.iou, 'device': a.device, 'detection_count': len(detections), 'python': platform.python_version(), 'torch': torch.__version__, 'ultralytics': ultralytics.__version__}
    (a.output_dir / 'bus_reference_meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    cv2.imwrite(str(a.output_dir / 'bus_reference.jpg'), result.plot())
    print(f'reference: {path} ({len(detections)} detections)'); return 0

def np_clip(values, offset, scale, limit):
    import numpy as np
    return np.clip((values - offset) / scale, 0, limit)

if __name__ == '__main__': raise SystemExit(main())
