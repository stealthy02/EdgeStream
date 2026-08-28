"""Shared YOLO preprocessing, output decoding, and small file helpers."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def letterbox(image: np.ndarray, target_size: int) -> tuple[np.ndarray, float, int, int]:
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = target_size - resized.shape[1], target_size - resized.shape[0]
    left, top = round(pad_w / 2 - 0.1), round(pad_h / 2 - 0.1)
    output = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return output, scale, left, top


def build_input_tensor(image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None], dtype=np.float32) / 255.0


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    result = np.empty_like(boxes)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1, y1 = np.maximum(box[:2], boxes[:, :2]).T
    x2, y2 = np.minimum(box[2:], boxes[:, 2:]).T
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    other = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    union = area + other - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold: float) -> np.ndarray:
    kept: list[int] = []
    for class_id in np.unique(classes):
        order = np.where(classes == class_id)[0]
        order = order[np.argsort(scores[order])[::-1]]
        while order.size:
            current = int(order[0]); kept.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            order = rest[_iou(boxes[current], boxes[rest]) <= iou_threshold]
    return np.asarray(sorted(kept, key=lambda i: float(scores[i]), reverse=True), dtype=np.int64)


def decode_outputs(outputs: list[np.ndarray] | tuple[np.ndarray, ...], conf: float, iou: float,
                   scale: float, pad_left: int, pad_top: int, width: int, height: int):
    """Decode either the original [1,84,N] output or split [boxes,scores] outputs."""

    def as_candidates(tensor: np.ndarray, field_count: int | None = None) -> np.ndarray:
        """Normalize [1,C,N] and [1,N,C] tensors to [N,C]."""
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise RuntimeError(f"Expected rank-3 batch-1 output, got {tensor.shape}")
        values = tensor[0]
        if field_count is not None:
            if values.shape[0] == field_count:
                return values.T
            if values.shape[1] == field_count:
                return values
            raise RuntimeError(f"Expected one dimension to equal {field_count}, got {tensor.shape}")
        # YOLO has far fewer fields than candidate points. The smaller non-batch
        # dimension is therefore the field dimension for the standard export.
        return values.T if values.shape[0] < values.shape[1] else values

    if len(outputs) == 1:
        raw = np.asarray(outputs[0])
        if raw.ndim != 3 or raw.shape[0] != 1:
            raise RuntimeError(f"Expected output [1,C,N], got {raw.shape}")
        predictions = as_candidates(raw)
        if predictions.shape[1] < 5:
            raise RuntimeError(f"Expected at least 5 output fields, got {raw.shape}")
        boxes_xywh, scores_matrix = predictions[:, :4], predictions[:, 4:]
    elif len(outputs) == 2:
        boxes, scores = (np.asarray(v) for v in outputs)
        if boxes.ndim != 3 or scores.ndim != 3 or boxes.shape[0] != 1 or scores.shape[0] != 1:
            raise RuntimeError(f"Expected split outputs [1,4,N] and [1,C,N], got {boxes.shape}, {scores.shape}")
        boxes_xywh = as_candidates(boxes, field_count=4)
        scores_matrix = as_candidates(scores)
    else:
        raise RuntimeError(f"Expected one or two outputs, got {len(outputs)}")
    if boxes_xywh.shape[0] != scores_matrix.shape[0]:
        raise RuntimeError("Box and score candidate counts differ")
    classes = np.argmax(scores_matrix, axis=1)
    scores = scores_matrix[np.arange(scores_matrix.shape[0]), classes]
    mask = scores >= conf
    boxes = xywh_to_xyxy(boxes_xywh[mask]); scores = scores[mask]; classes = classes[mask]
    keep = _nms(boxes, scores, classes, iou)
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    if boxes.size:
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad_left) / scale, 0, width)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad_top) / scale, 0, height)
    return boxes, scores, classes


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def compare_detections(reference: list[dict[str, Any]],
                       actual: list[dict[str, Any]],
                       score_atol: float,
                       box_atol: float) -> dict[str, Any]:
    """逐框贪心匹配（同类别 + 最小 L2 距离），在容差内判定通过。

    用途：ONNX 导出验证（pass/fail）。
    精度报告（precision/recall/IoU/类别统计）请用 scripts/accuracy_benchmark.py
    中的 match_and_score —— 两者算法与目标不同，故各自独立实现。
    """
    unmatched: set[int] = set(range(len(reference)))
    matches: list[dict[str, Any]] = []
    passed = len(reference) == len(actual)

    for ai, item in enumerate(actual):
        candidates = [i for i in unmatched
                      if int(reference[i]["class_id"]) == item["class_id"]]
        if not candidates:
            passed = False
            matches.append({"actual_index": ai, "matched": False})
            continue
        ri = min(candidates,
                 key=lambda i: sum((item["xyxy"][j] - reference[i]["xyxy"][j]) ** 2
                                   for j in range(4)))
        unmatched.remove(ri)
        score_error = abs(item["confidence"] - float(reference[ri]["confidence"]))
        box_error = max(abs(item["xyxy"][j] - float(reference[ri]["xyxy"][j]))
                        for j in range(4))
        ok = score_error <= score_atol and box_error <= box_atol
        passed &= ok
        matches.append({
            "actual_index": ai, "reference_index": ri,
            "score_abs_error": score_error,
            "box_max_abs_error": box_error,
            "passed": ok,
        })
    passed &= not unmatched
    return {
        "passed": passed,
        "reference_count": len(reference),
        "actual_count": len(actual),
        "unmatched_reference_indices": sorted(unmatched),
        "reference_class_counts": dict(
            Counter(x.get("class_name", f"class_{x['class_id']}") for x in reference)),
        "actual_class_counts": dict(
            Counter(x.get("class_name", f"class_{x['class_id']}") for x in actual)),
        "matches": matches,
    }

