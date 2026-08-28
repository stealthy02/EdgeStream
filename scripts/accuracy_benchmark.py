#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【阶段 3 / 3】纯离线精度对比报告 —— 完全不做推理，只负责读 JSONL 出报告。

⚠️  本脚本不做任何模型推理，因此 100% 不会干扰你验证 C++ 代码的正确性。
    它只是一个"批改卷子"的角色：
      · 参考"标准答案" = 阶段 1 generate_reference.py 跑的 pt.jsonl  (PyTorch 原生)
      · "待批卷子"   = 阶段 2 bus_test (C++) 跑的 onnx / rknn_fp16 / rknn_int8 .jsonl

输入：阶段 1 + 阶段 2 共同写入的 run_dir，目录结构必须是：
  artifacts/accuracy_eval/{run_id}/
    ├── {image_stem_1}/
    │   ├── pt.jsonl           （阶段 1，参考基准）
    │   ├── onnx.jsonl         （阶段 2 C++ ONNX 推理结果，待验证）
    │   ├── rknn_fp16.jsonl    （阶段 2 C++ RKNN FP16 推理结果，待验证）
    │   └── rknn_int8.jsonl    （阶段 2 C++ RKNN INT8 推理结果，待验证）
    ├── {image_stem_2}/
    ├── ...
    └── manifest.json          （阶段 1 写入，可选，缺失时自动扫目录）

用法：
  python3 scripts/accuracy_benchmark.py --run-dir artifacts/accuracy_eval/20260828_120000
  python3 scripts/accuracy_benchmark.py --run-dir ... --match-iou 0.5   # 改匹配 IoU 阈值
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ============================================================================
# 轻量 JSONL 读写（不再依赖 yolo_pipeline，脚本可独立放到任意机器运行）
# ============================================================================
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# ============================================================================
# 匹配与评分算法（贪心：同类别 + 最大 IoU ≥ 阈值 为一对）
# ============================================================================
def iou_xyxy(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def match_and_score(reference: list[dict[str, Any]],
                    actual: list[dict[str, Any]],
                    iou_threshold: float = 0.5) -> dict[str, Any]:
    unmatched_ref = list(range(len(reference)))
    unmatched_act = list(range(len(actual)))
    matches: list[dict[str, Any]] = []

    for ri in list(unmatched_ref):
        r = reference[ri]
        best_iou = 0.0; best_ai = -1
        for ai in unmatched_act:
            a = actual[ai]
            if a.get("class_id", -1) != r.get("class_id", -2):
                continue
            i = iou_xyxy(r["xyxy"], a["xyxy"])
            if i > best_iou:
                best_iou = i; best_ai = ai
        if best_ai >= 0 and best_iou >= iou_threshold:
            conf_diff = abs(float(r["confidence"]) - float(actual[best_ai]["confidence"]))
            matches.append({
                "reference_index": ri,
                "actual_index": best_ai,
                "class_id": r["class_id"],
                "class_name": r.get("class_name",
                                    actual[best_ai].get("class_name",
                                        f"class_{r['class_id']}")),
                "iou": float(best_iou),
                "confidence_ref": float(r["confidence"]),
                "confidence_act": float(actual[best_ai]["confidence"]),
                "confidence_diff": float(conf_diff),
            })
            unmatched_ref.remove(ri)
            unmatched_act.remove(best_ai)

    tp, fn, fp = len(matches), len(unmatched_ref), len(unmatched_act)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    avg_iou = float(sum(m["iou"] for m in matches) / tp) if tp else 0.0
    avg_cd  = float(sum(m["confidence_diff"] for m in matches) / tp) if tp else 0.0

    ref_by_cls = Counter(r.get("class_name", f"class_{r['class_id']}") for r in reference)
    act_by_cls = Counter(a.get("class_name", f"class_{a['class_id']}") for a in actual)
    by_cls_matches: dict[str, list[dict]] = defaultdict(list)
    for m in matches:
        by_cls_matches[m["class_name"]].append(m)
    all_cls = set(list(ref_by_cls.keys()) + list(act_by_cls.keys()) + list(by_cls_matches.keys()))
    class_stats = {}
    for cls in all_cls:
        cm = by_cls_matches[cls]
        cre = ref_by_cls.get(cls, 0); cac = act_by_cls.get(cls, 0); ctp = len(cm)
        class_stats[cls] = {
            "reference_count": cre, "actual_count": cac, "true_positive": ctp,
            "fn": cre - ctp, "fp": cac - ctp,
            "recall": (ctp / cre) if cre > 0 else 0.0,
            "precision": (ctp / cac) if cac > 0 else 0.0,
            "avg_iou": float(sum(m["iou"] for m in cm) / ctp) if ctp else 0.0,
            "avg_conf_diff": float(sum(m["confidence_diff"] for m in cm) / ctp) if ctp else 0.0,
        }
    missed = [{"class_name": reference[i].get("class_name", f"class_{reference[i]['class_id']}"),
               "class_id": reference[i]["class_id"],
               "confidence": reference[i]["confidence"],
               "xyxy": reference[i]["xyxy"]} for i in unmatched_ref]
    falsep = [{"class_name": actual[i].get("class_name", f"class_{actual[i]['class_id']}"),
               "class_id": actual[i]["class_id"],
               "confidence": actual[i]["confidence"],
               "xyxy": actual[i]["xyxy"]} for i in unmatched_act]

    return {
        "iou_threshold": iou_threshold,
        "reference_count": len(reference), "actual_count": len(actual),
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "avg_iou": avg_iou, "avg_confidence_diff": avg_cd,
        "class_stats": class_stats,
        "matches": matches,
        "missed_detections": missed,
        "false_positive_detections": falsep,
    }

# ============================================================================
# 汇总与报告生成
# ============================================================================
STAGES = ("pt", "onnx", "rknn_fp16", "rknn_int8")
STAGE_LABELS = {
    "pt": "PyTorch (参考)", "onnx": "ONNX (C++)",
    "rknn_fp16": "RKNN FP16 (C++)", "rknn_int8": "RKNN INT8 (C++)",
}

def build_summary(run_dir: Path, image_names: list[str],
                  per_image: dict[str, dict[str, Any]],
                  stages_present: list[str],
                  match_iou: float) -> tuple[dict[str, Any], str]:
    summary = {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "env": {
            "python": platform.python_version(),
            "system": f"{platform.system()} {platform.release()}",
            "note": "本脚本纯离线，未做任何推理。检测结果来源请查看 per_image.*.source。",
        },
        "match_iou_threshold": match_iou,
        "stages_present": stages_present,
        "image_count": len(image_names),
        "images": image_names,
        "per_image": per_image,
    }

    # 全局聚合
    overall: dict[str, Any] = {}
    for stage in [s for s in STAGES if s != "pt" and s in stages_present]:
        tps = fps = fns = 0
        ious, cds, infer_times = [], [], []
        cls_stat: dict[str, dict[str, float]] = defaultdict(lambda: {
            "reference_count": 0, "actual_count": 0, "tp": 0, "fn": 0, "fp": 0,
            "iou_sum": 0.0, "cd_sum": 0.0,
        })
        for img in image_names:
            comp = per_image[img]["comparisons"].get(stage)
            if not comp or "true_positive" not in comp:
                continue
            tps += comp["true_positive"]; fps += comp["false_positive"]; fns += comp["false_negative"]
            for m in comp["matches"]:
                ious.append(m["iou"]); cds.append(m["confidence_diff"])
            if "infer_ms" in per_image[img].get("meta", {}).get(stage, {}):
                infer_times.append(per_image[img]["meta"][stage]["infer_ms"])
            for cls, cs in comp["class_stats"].items():
                c = cls_stat[cls]
                c["reference_count"] += cs["reference_count"]
                c["actual_count"]    += cs["actual_count"]
                c["tp"]              += cs["true_positive"]
                c["fn"]              += cs.get("fn", cs["reference_count"] - cs["true_positive"])
                c["fp"]              += cs.get("fp", cs["actual_count"] - cs["true_positive"])
                c["iou_sum"]         += cs["avg_iou"] * cs["true_positive"]
                c["cd_sum"]          += cs["avg_conf_diff"] * cs["true_positive"]
        prec = tps / (tps + fps) if (tps + fps) > 0 else 0.0
        rec  = tps / (tps + fns) if (tps + fns) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        avg_iou = float(sum(ious) / len(ious)) if ious else 0.0
        avg_cd  = float(sum(cds)  / len(cds))  if cds  else 0.0
        avg_inf = float(sum(infer_times) / len(infer_times)) if infer_times else 0.0
        cls_out = {}
        for cls, c in cls_stat.items():
            tp_c = c["tp"]
            cls_out[cls] = {
                "reference_count": c["reference_count"], "actual_count": c["actual_count"],
                "true_positive": tp_c, "fn": c["fn"], "fp": c["fp"],
                "recall":    (tp_c / c["reference_count"]) if c["reference_count"] > 0 else 0.0,
                "precision": (tp_c / c["actual_count"])    if c["actual_count"]    > 0 else 0.0,
                "avg_iou":         (c["iou_sum"] / tp_c) if tp_c > 0 else 0.0,
                "avg_conf_diff":   (c["cd_sum"]  / tp_c) if tp_c > 0 else 0.0,
            }
        overall[stage] = {
            "true_positive": tps, "false_positive": fps, "false_negative": fns,
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "avg_iou": avg_iou, "avg_confidence_diff": avg_cd, "avg_infer_ms": avg_inf,
            "class_stats": cls_out,
        }
    summary["overall"] = overall

    # ============ Markdown ============
    md: list[str] = []
    md.append(f"# YOLO 四阶段精度对比报告 —— `{run_dir.name}`")
    md.append("")
    md.append(f"- 生成时间：{summary['generated_at']}")
    md.append(f"- 匹配 IoU 阈值：≥ {match_iou}")
    md.append(f"- 图片数量：{len(image_names)}")
    md.append(f"- 存在阶段：{' / '.join(f'{STAGE_LABELS[s]} ({s})' for s in stages_present)}")
    if "pt" not in stages_present:
        md.append("- ⚠️  **缺少 PT 参考 JSONL** — 以下对比指标均以 ONNX 为基准（不推荐），请先执行阶段 1。")
    md.append("")
    md.append("> 本报告基于 C++ 推理输出的 JSONL 生成，100% 离线，未做任何推理，用于验证 C++ 代码 (preprocess → inference_engine/RknnEngine → postprocess) 正确性。")
    md.append("")

    # Table 1：全局汇总
    md.append("## 1. 全局汇总指标（PT 为参考基准）")
    md.append("")
    md.append("| 阶段 (C++) | 总检出 | TP | FP | FN | Precision | Recall | F1 | Avg IoU | Avg ConfΔ |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for stage in [s for s in ("onnx", "rknn_fp16", "rknn_int8") if s in overall]:
        o = overall[stage]
        total = o["true_positive"] + o["false_positive"]
        md.append(
            f"| {STAGE_LABELS[stage]} | {total} "
            f"| {o['true_positive']} | {o['false_positive']} | {o['false_negative']} "
            f"| {o['precision']:.4f} | {o['recall']:.4f} | {o['f1']:.4f} "
            f"| {o['avg_iou']:.4f} | {o['avg_confidence_diff']:.4f} |"
        )
    md.append("")

    # Table 2：逐图 Recall / IoU
    candidate_stages = [s for s in ("onnx", "rknn_fp16", "rknn_int8") if s in stages_present]
    if candidate_stages:
        md.append("## 2. 逐图对比")
        md.append("")
        cols = ["图片", "PT 框数"]
        for s in candidate_stages: cols += [f"{s} R", f"{s} mIoU"]
        md.append("| " + " | ".join(cols) + " |")
        md.append("|" + "|".join(["---"] * len(cols)) + "|")
        for img in image_names:
            info = per_image[img]
            pt_count = int(info["detections"].get("pt", 0))
            row = [img, str(pt_count)]
            for s in candidate_stages:
                comp = info["comparisons"].get(s)
                if comp and "recall" in comp:
                    row.append(f"{comp['recall']:.3f}")
                    row.append(f"{comp['avg_iou']:.3f}")
                else:
                    row += ["-", "-"]
            md.append("| " + " | ".join(row) + " |")
        md.append("")

    # Table 3：RKNN INT8 类别级掉点明细（重灾区）
    if "rknn_int8" in overall:
        md.append("## 3. RKNN INT8 类别级掉点明细（与 PT 参考比对）")
        md.append("")
        md.append("| 类别 | 参考框 | INT8检出 | TP | 漏检 | 误检 | Recall | Precision | Avg IoU | Avg ConfΔ |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        cs = overall["rknn_int8"]["class_stats"]
        # 掉点严重的在前（recall 低 + 参考框数 > 0 的优先展示）
        def sort_key(item):
            cls, c = item
            has_ref = 1 if c["reference_count"] > 0 else 0
            return (-has_ref, c["recall"], -c["reference_count"])
        for cls, c in sorted(cs.items(), key=sort_key):
            md.append(
                f"| {cls} | {c['reference_count']} | {c['actual_count']} "
                f"| {c['true_positive']} | {c['fn']} | {c['fp']} "
                f"| {c['recall']:.3f} | {c['precision']:.3f} "
                f"| {c['avg_iou']:.3f} | {c['avg_conf_diff']:.4f} |"
            )
        md.append("")

        # 额外的诊断：INT8 漏检 / 误检 Top 图片
        md.append("## 4. RKNN INT8 问题定位（漏检误检最多的 Top 图片）")
        md.append("")
        ranked = []
        for img in image_names:
            comp = per_image[img]["comparisons"].get("rknn_int8")
            if not comp or "false_negative" not in comp:
                continue
            ranked.append((img, comp["false_negative"] + comp["false_positive"],
                           comp["false_negative"], comp["false_positive"],
                           comp.get("missed_detections", []),
                           comp.get("false_positive_detections", [])))
        ranked.sort(key=lambda x: -x[1])
        for img, total, fn, fp, missed, fp_det in ranked[:5]:
            if total == 0: continue
            md.append(f"### {img}  (漏 {fn} / 误 {fp})")
            if missed:
                names = Counter(m["class_name"] for m in missed).most_common()
                md.append(f"- 漏检类别分布：{', '.join(f'{n}×{c}' for n,c in names)}")
            if fp_det:
                names = Counter(m["class_name"] for m in fp_det).most_common()
                md.append(f"- 误检类别分布：{', '.join(f'{n}×{c}' for n,c in names)}")
            md.append("")

    return summary, "\n".join(md) + "\n"


# ============================================================================
# 主流程：扫目录 → 读 JSONL → 算指标 → 写报告
# ============================================================================
def main() -> int:
    p = argparse.ArgumentParser(description="【阶段 3/3】纯离线精度对比报告 (不做推理)")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="阶段 1 + 阶段 2 共同产出的目录 (含 {stem}/*.jsonl)")
    p.add_argument("--match-iou", type=float, default=0.5,
                   help="参考框与预测框匹配所需最小 IoU (默认 0.5)")
    args = p.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir 不存在: {run_dir}")

    # 1) 收集每张图的子目录（包含任意 jsonl 就算）
    image_dirs: list[Path] = sorted([
        d for d in run_dir.iterdir() if d.is_dir()
        and any(p.suffix == ".jsonl" for p in d.iterdir())
    ])
    if not image_dirs:
        raise RuntimeError(f"run_dir 下没有找到任何图片子目录 + jsonl：{run_dir}")

    # 2) 加载 manifest（如果有阶段 1 写的）
    manifest: dict[str, Any] = {}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ordered_names = manifest.get("images", [])
        # 按 manifest 中的顺序排（保证和阶段 1 一致），再追加那些 manifest 没有但目录里有的
        stem_to_dir = {d.name: d for d in image_dirs}
        ordered_dirs = [stem_to_dir[n] for n in ordered_names if n in stem_to_dir]
        for d in image_dirs:
            if d not in ordered_dirs:
                ordered_dirs.append(d)
        image_dirs = ordered_dirs

    print(f"【阶段 3/3】纯离线精度对比")
    print(f"  run_dir   = {run_dir}")
    print(f"  match IoU = {args.match_iou}")
    print(f"  图片目录数 = {len(image_dirs)}\n")

    # 3) 逐图读 4 份 JSONL + 算对比
    per_image: dict[str, dict[str, Any]] = {}
    image_names: list[str] = []
    stages_present_set: set[str] = set()
    for idx, d in enumerate(image_dirs, 1):
        stem = d.name
        image_names.append(stem)
        print(f"  [{idx}/{len(image_dirs)}] {stem} ... ", end="", flush=True)
        dets: dict[str, list[dict]] = {}
        meta: dict[str, dict] = {}
        for stage in STAGES:
            jsonl_path = d / f"{stage}.jsonl"
            items = load_jsonl(jsonl_path)
            if items:
                dets[stage] = items
                stages_present_set.add(stage)
                meta[stage] = {"source": str(jsonl_path.name), "detection_count": len(items)}
        comparisons: dict[str, dict] = {}
        ref = dets.get("pt", [])
        for stage in STAGES:
            if stage == "pt" or stage not in dets:
                continue
            if not ref:
                comparisons[stage] = {"skipped": "pt reference missing"}
                continue
            comparisons[stage] = match_and_score(ref, dets[stage], iou_threshold=args.match_iou)
        # 存每张图的 comparison.json（可独立审查）
        (d / "comparison.json").write_text(
            json.dumps({"detection_counts": {s: len(v) for s, v in dets.items()},
                        "meta": meta, "comparisons": comparisons},
                       indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        per_image[stem] = {"detections": {s: len(v) for s, v in dets.items()},
                           "meta": meta, "comparisons": comparisons}
        # 一行摘要：哪些阶段有数据 + 每个阶段 vs PT 的 Recall
        if ref:
            parts = []
            for s in ("onnx", "rknn_fp16", "rknn_int8"):
                if s in comparisons and "recall" in comparisons[s]:
                    parts.append(f"{s} R={comparisons[s]['recall']:.2f} IoU={comparisons[s]['avg_iou']:.2f}")
            print("OK   " + " | ".join(parts))
        else:
            print("OK   (无 PT 参考，跳过对比)")

    stages_present = [s for s in STAGES if s in stages_present_set]
    print()

    # 4) 生成汇总
    summary, md = build_summary(run_dir, image_names, per_image, stages_present, args.match_iou)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    print(f"✅ 报告生成完毕")
    print(f"  📊 summary.json → {run_dir / 'summary.json'}")
    print(f"  📝 summary.md   → {run_dir / 'summary.md'}")
    print()
    print("【提示】重点查看 summary.md 里的 3 张表格：")
    print("  1. 全局汇总：一眼看出 C++ 链路哪个阶段掉点最重")
    print("  2. 逐图对比：定位是某几张特殊图的问题还是普遍问题")
    print("  3. RKNN INT8 类别掉点：哪个类 Recall 低 → 反查 C++ 后处理或量化校准集")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
