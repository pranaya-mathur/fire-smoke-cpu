#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
(ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / ".cache" / "ultralytics").mkdir(parents=True, exist_ok=True)


def iter_videos(path: Path):
    if path.is_file():
        yield path
    elif path.exists():
        for suffix in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.m4v"):
            yield from sorted(path.rglob(suffix))


def load_manifest(path: Path | None) -> dict[str, dict]:
    if not path:
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        return {str(Path(row["video_path"]).resolve()): row for row in csv.DictReader(f)}


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def infer_ground_truth(video_path: Path) -> dict:
    name = f"{video_path.parent.name} {video_path.name}".lower()
    is_negative = any(token in name for token in ("negative", "normal", "no_fire", "nofire"))
    if is_negative:
        return {"has_fire": False, "has_smoke": False, "is_negative": True, "source": "filename_inference"}
    return {
        "has_fire": "fire" in name,
        "has_smoke": "smoke" in name,
        "is_negative": False,
        "unknown": "fire" not in name and "smoke" not in name,
        "source": "filename_inference",
    }


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--inference-fps", type=float, default=3.0)
    parser.add_argument("--fire-threshold", type=float, default=0.30)
    parser.add_argument("--smoke-threshold", type=float, default=0.30)
    parser.add_argument("--fire-window-seconds", type=float, default=5.0)
    parser.add_argument("--fire-min-detections", type=int, default=3)
    parser.add_argument("--smoke-persistence-seconds", type=float, default=4.0)
    parser.add_argument("--smoke-iou-threshold", type=float, default=0.10)
    parser.add_argument("--cooldown-seconds", type=float, default=10.0)
    parser.add_argument("--infer-ground-truth-from-filename", action="store_true")
    args = parser.parse_args()

    videos = list(iter_videos(args.source))
    manifest = load_manifest(args.manifest)
    out_json = ROOT / "reports/local_video_evaluation.json"
    out_md = ROOT / "reports/local_video_evaluation.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if not videos:
        payload = {"status": "READY_NO_LOCAL_VIDEOS", "source": str(args.source), "video_reports": []}
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        out_md.write_text("# Local Video Evaluation\n\nNo local MP4/MOV/AVI/MKV files were found. Tooling is ready; no results were invented.\n", encoding="utf-8")
        print(f"Wrote {out_json}")
        return 0

    import cv2
    from ultralytics import YOLO

    model = YOLO(args.model)
    reports = []
    totals = {"candidate_alerts": 0, "confirmed_alerts": 0, "false_confirmed_alerts": 0, "missed_events": 0}

    for video in videos:
        resolved = str(video.resolve())
        gt_row = manifest.get(resolved, {})
        if gt_row:
            gt = {"has_fire": truthy(gt_row.get("has_fire")), "has_smoke": truthy(gt_row.get("has_smoke")), "is_negative": truthy(gt_row.get("is_negative")), "source": "manifest"}
        elif args.infer_ground_truth_from_filename:
            gt = infer_ground_truth(video)
        else:
            gt = {"unknown": True, "source": "not_provided"}

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps else 0.0
        stride = max(1, round(fps / args.inference_fps)) if fps and args.inference_fps > 0 else 1
        frame_idx = 0
        inference_frames = 0
        latencies = []
        first_detection_time = None
        first_confirmed_time = None
        last_alert_time = -10**9
        fire_window = deque()
        smoke_tracks = deque()
        candidate_alerts = []
        confirmed_alerts = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if (frame_idx - 1) % stride != 0:
                continue
            timestamp = frame_idx / fps if fps else float(inference_frames)
            inference_frames += 1
            start = time.perf_counter()
            result = model.predict(frame, imgsz=args.imgsz, device=args.device, conf=min(args.fire_threshold, args.smoke_threshold), verbose=False)[0]
            latencies.append(time.perf_counter() - start)
            classes = [int(c) for c in result.boxes.cls.cpu().tolist()] if result.boxes is not None else []
            confs = [float(c) for c in result.boxes.conf.cpu().tolist()] if result.boxes is not None else []
            boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
            fire_hit = any(cls == 0 and conf >= args.fire_threshold for cls, conf in zip(classes, confs))
            smoke_boxes = [box for cls, conf, box in zip(classes, confs, boxes) if cls == 1 and conf >= args.smoke_threshold]
            smoke_hit = bool(smoke_boxes)
            if (fire_hit or smoke_hit) and first_detection_time is None:
                first_detection_time = timestamp

            fire_window.append((timestamp, fire_hit))
            while fire_window and timestamp - fire_window[0][0] > args.fire_window_seconds:
                fire_window.popleft()
            if sum(1 for _, hit in fire_window if hit) >= args.fire_min_detections:
                candidate_alerts.append({"timestamp": timestamp, "type": "fire_candidate"})
                if timestamp - last_alert_time >= args.cooldown_seconds:
                    confirmed_alerts.append({"timestamp": timestamp, "type": "fire"})
                    first_confirmed_time = first_confirmed_time or timestamp
                    last_alert_time = timestamp
                fire_window.clear()

            if smoke_hit:
                consistent = not smoke_tracks or any(iou_xyxy(prev_box, box) >= args.smoke_iou_threshold for _, prev_box in smoke_tracks for box in smoke_boxes)
                if consistent:
                    for box in smoke_boxes:
                        smoke_tracks.append((timestamp, box))
            while smoke_tracks and timestamp - smoke_tracks[0][0] > args.smoke_persistence_seconds:
                smoke_tracks.popleft()
            if smoke_tracks and timestamp - smoke_tracks[0][0] >= args.smoke_persistence_seconds:
                candidate_alerts.append({"timestamp": timestamp, "type": "smoke_candidate"})
                if timestamp - last_alert_time >= args.cooldown_seconds:
                    confirmed_alerts.append({"timestamp": timestamp, "type": "smoke"})
                    first_confirmed_time = first_confirmed_time or timestamp
                    last_alert_time = timestamp
                smoke_tracks.clear()
        cap.release()

        has_confirmed_fire = any(a["type"] == "fire" for a in confirmed_alerts)
        has_confirmed_smoke = any(a["type"] == "smoke" for a in confirmed_alerts)
        evaluable = not gt.get("unknown", False)
        missed = 0
        false_confirmed = 0
        if evaluable:
            missed += int(gt.get("has_fire") and not has_confirmed_fire)
            missed += int(gt.get("has_smoke") and not has_confirmed_smoke)
            false_confirmed += sum(1 for a in confirmed_alerts if a["type"] == "fire" and not gt.get("has_fire"))
            false_confirmed += sum(1 for a in confirmed_alerts if a["type"] == "smoke" and not gt.get("has_smoke"))
        for key, val in {"candidate_alerts": len(candidate_alerts), "confirmed_alerts": len(confirmed_alerts), "false_confirmed_alerts": false_confirmed, "missed_events": missed}.items():
            totals[key] += val
        reports.append(
            {
                "video_path": str(video),
                "ground_truth": gt,
                "duration_seconds": duration,
                "native_fps": fps,
                "inference_frames": inference_frames,
                "effective_ai_fps": inference_frames / duration if duration else None,
                "first_detection_time": first_detection_time,
                "first_confirmed_alert_time": first_confirmed_time,
                "event_recall": None if not evaluable else (1.0 if missed == 0 else 0.0),
                "missed_events": missed,
                "candidate_alerts": candidate_alerts,
                "confirmed_alerts": confirmed_alerts,
                "false_candidate_alerts": None,
                "false_confirmed_alerts": false_confirmed,
                "false_alerts_per_video_hour": false_confirmed / (duration / 3600) if duration else None,
                "avg_latency_ms": statistics.mean(latencies) * 1000 if latencies else None,
                "p50_latency_ms": statistics.median(latencies) * 1000 if latencies else None,
                "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95) - 1] * 1000 if latencies else None,
            }
        )

    payload = {"status": "COMPLETED", "settings": vars(args) | {"source": str(args.source), "manifest": str(args.manifest) if args.manifest else ""}, "summary": totals | {"total_videos": len(reports)}, "video_reports": reports}
    out_json.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    out_md.write_text("# Local Video Evaluation\n\n" + json.dumps(payload["summary"], indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
