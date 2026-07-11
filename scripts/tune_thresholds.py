#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
(ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / ".cache" / "ultralytics").mkdir(parents=True, exist_ok=True)

from fire_smoke_cpu.annotations import YoloBox, parse_yolo_label
from fire_smoke_cpu.constants import CLASS_ID_TO_NAME, MANIFEST_DIR, REPORT_DIR


@dataclass(frozen=True)
class PredBox:
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def yolo_to_xyxy(box: YoloBox) -> tuple[float, float, float, float]:
    half_w = box.width / 2
    half_h = box.height / 2
    return box.x_center - half_w, box.y_center - half_h, box.x_center + half_w, box.y_center + half_h


def iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - inter
    return inter / union if union > 0 else 0.0


def evaluate_threshold(rows: list[dict], predictions: dict[str, list[PredBox]], threshold: float, iou_threshold: float) -> dict:
    per_class = {
        class_id: {"tp": 0, "fp": 0, "fn": 0, "images_with_prediction": 0, "images_with_label": 0}
        for class_id in CLASS_ID_TO_NAME
    }
    for row in rows:
        labels, _errors = parse_yolo_label(Path(row["canonical_label_path"]))
        labels_by_class: dict[int, list[tuple[float, float, float, float]]] = {0: [], 1: []}
        for label in labels:
            labels_by_class[label.class_id].append(yolo_to_xyxy(label))
        image_predictions = [box for box in predictions[row["sample_id"]] if box.confidence >= threshold]
        for class_id in CLASS_ID_TO_NAME:
            gt = labels_by_class[class_id]
            preds = [box for box in image_predictions if box.class_id == class_id]
            per_class[class_id]["images_with_label"] += int(bool(gt))
            per_class[class_id]["images_with_prediction"] += int(bool(preds))
            matched: set[int] = set()
            for pred in sorted(preds, key=lambda item: item.confidence, reverse=True):
                best_idx = -1
                best_iou = 0.0
                pred_box = (pred.x1, pred.y1, pred.x2, pred.y2)
                for idx, gt_box in enumerate(gt):
                    if idx in matched:
                        continue
                    score = iou(pred_box, gt_box)
                    if score > best_iou:
                        best_iou = score
                        best_idx = idx
                if best_idx >= 0 and best_iou >= iou_threshold:
                    matched.add(best_idx)
                    per_class[class_id]["tp"] += 1
                else:
                    per_class[class_id]["fp"] += 1
            per_class[class_id]["fn"] += max(0, len(gt) - len(matched))
    metrics = {"threshold": threshold, "iou_threshold": iou_threshold, "images": len(rows), "classes": {}}
    for class_id, counts in per_class.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics["classes"][CLASS_ID_TO_NAME[class_id]] = {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives_per_image": fp / max(len(rows), 1),
        }
    return metrics


def choose_threshold(metrics: list[dict], class_name: str, min_precision: float, min_recall: float) -> dict:
    candidates = []
    for row in metrics:
        cls = row["classes"][class_name]
        if cls["precision"] >= min_precision and cls["recall"] >= min_recall:
            candidates.append(row)
    if candidates:
        return max(candidates, key=lambda item: (item["classes"][class_name]["f1"], item["classes"][class_name]["recall"]))
    return max(metrics, key=lambda item: (item["classes"][class_name]["f1"], item["classes"][class_name]["recall"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "all_samples.csv")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--min-fire-precision", type=float, default=0.70)
    parser.add_argument("--min-fire-recall", type=float, default=0.30)
    parser.add_argument("--min-smoke-precision", type=float, default=0.35)
    parser.add_argument("--min-smoke-recall", type=float, default=0.25)
    args = parser.parse_args()

    from ultralytics import YOLO

    rows = [
        row
        for row in csv.DictReader(args.manifest.open("r", encoding="utf-8"))
        if row.get("split") == args.split and row.get("excluded") not in {"True", "true", "1"}
    ]
    if not rows:
        print(f"No rows found for split={args.split} in {args.manifest}")
        return 2

    model = YOLO(args.model)
    predictions: dict[str, list[PredBox]] = {}
    for row in rows:
        result = model.predict(
            source=row["canonical_image_path"],
            device=args.device,
            imgsz=args.imgsz,
            conf=0.001,
            iou=0.7,
            verbose=False,
        )[0]
        boxes: list[PredBox] = []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxyn.cpu().tolist()
            cls = result.boxes.cls.cpu().tolist()
            conf = result.boxes.conf.cpu().tolist()
            boxes = [
                PredBox(int(class_id), float(score), float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))
                for coords, class_id, score in zip(xyxy, cls, conf, strict=False)
            ]
        predictions[row["sample_id"]] = boxes

    thresholds = [float(value) for value in args.thresholds.split(",")]
    metrics = [evaluate_threshold(rows, predictions, threshold, args.iou) for threshold in thresholds]
    recommendation = {
        "fire": choose_threshold(metrics, "fire", args.min_fire_precision, args.min_fire_recall)["threshold"],
        "smoke": choose_threshold(metrics, "smoke", args.min_smoke_precision, args.min_smoke_recall)["threshold"],
    }
    report = {
        "model": args.model,
        "split": args.split,
        "image_count": len(rows),
        "iou_threshold": args.iou,
        "thresholds": metrics,
        "recommended_thresholds": recommendation,
        "note": "False positives are unmatched predictions on labeled validation/test images, not a substitute for a dedicated negative CCTV validation set.",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORT_DIR / f"threshold_tuning_{args.split}.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = [
        f"# Threshold Tuning ({args.split})",
        "",
        f"- Model: `{args.model}`",
        f"- Images: `{len(rows)}`",
        f"- IoU match threshold: `{args.iou}`",
        f"- Recommended fire threshold: `{recommendation['fire']}`",
        f"- Recommended smoke threshold: `{recommendation['smoke']}`",
        "",
        "| Threshold | Fire precision | Fire recall | Fire FP/img | Smoke precision | Smoke recall | Smoke FP/img |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        fire = row["classes"]["fire"]
        smoke = row["classes"]["smoke"]
        md.append(
            "| "
            + " | ".join(
                [
                    f"{row['threshold']:.2f}",
                    f"{fire['precision']:.3f}",
                    f"{fire['recall']:.3f}",
                    f"{fire['false_positives_per_image']:.3f}",
                    f"{smoke['precision']:.3f}",
                    f"{smoke['recall']:.3f}",
                    f"{smoke['false_positives_per_image']:.3f}",
                ]
            )
            + " |"
        )
    md.extend(
        [
            "",
            "False-positive rates here are unmatched predictions on labeled validation/test images. A real false-alarm gate still needs negative CCTV clips/images.",
        ]
    )
    out_md = REPORT_DIR / f"threshold_tuning_{args.split}.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
