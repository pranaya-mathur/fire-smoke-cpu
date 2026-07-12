#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    else:
        for suffix in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.m4v"):
            yield from sorted(path.rglob(suffix))


def infer_ground_truth(video_path: Path) -> dict:
    name = video_path.name.lower()
    parent = video_path.parent.name.lower()
    has_fire = "fire" in name or "fire" in parent
    has_smoke = "smoke" in name or "smoke" in parent
    is_negative = "negative" in name or "negative" in parent or "normal" in name
    
    if is_negative:
        return {"fire": False, "smoke": False}
    
    return {
        "fire": has_fire,
        "smoke": has_smoke,
        "unknown": not has_fire and not has_smoke
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--fire-n", type=int, default=3)
    parser.add_argument("--fire-window", type=int, default=5)
    parser.add_argument("--smoke-persistence-frames", type=int, default=10)
    args = parser.parse_args()
    import cv2
    from ultralytics import YOLO

    model = YOLO(args.model)
    all_reports = []
    
    total_tp = 0
    total_fp = 0
    
    for video in iter_videos(args.source):
        gt = infer_ground_truth(video)
        
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        latencies = []
        frame_idx = 0
        fire_window = deque(maxlen=args.fire_window)
        smoke_window = deque(maxlen=args.smoke_persistence_frames)
        fire_detections = smoke_detections = 0
        first_detection = None
        alerts = []
        
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            start = time.perf_counter()
            result = model.predict(frame, imgsz=args.imgsz, device=args.device, conf=args.conf, verbose=False)[0]
            latencies.append(time.perf_counter() - start)
            classes = [int(c) for c in result.boxes.cls.cpu().tolist()] if result.boxes is not None else []
            has_fire = 0 in classes
            has_smoke = 1 in classes
            fire_detections += int(has_fire)
            smoke_detections += int(has_smoke)
            
            if (has_fire or has_smoke) and first_detection is None:
                first_detection = frame_idx / fps if fps else None
                
            fire_window.append(has_fire)
            smoke_window.append(has_smoke)
            
            if sum(fire_window) >= args.fire_n:
                alerts.append({"frame": frame_idx, "timestamp": frame_idx / fps if fps else 0, "type": "fire_candidate"})
                fire_window.clear()
            if len(smoke_window) == smoke_window.maxlen and all(smoke_window):
                alerts.append({"frame": frame_idx, "timestamp": frame_idx / fps if fps else 0, "type": "smoke_candidate"})
                smoke_window.clear()
        cap.release()
        
        # Calculate TP / FP for this video based on alerts
        fire_alerts = len([a for a in alerts if a["type"] == "fire_candidate"])
        smoke_alerts = len([a for a in alerts if a["type"] == "smoke_candidate"])
        
        vid_tp = 0
        vid_fp = 0
        
        if gt.get("unknown", False):
            pass # Skip event evaluation for completely unknown videos
        else:
            if fire_alerts > 0:
                if gt.get("fire"): vid_tp += 1
                else: vid_fp += fire_alerts
            
            if smoke_alerts > 0:
                if gt.get("smoke"): vid_tp += 1
                else: vid_fp += smoke_alerts
                
        total_tp += vid_tp
        total_fp += vid_fp
        
        all_reports.append(
            {
                "video": str(video),
                "ground_truth": gt,
                "total_frames": total,
                "processed_frames": frame_idx,
                "duration_seconds": total / fps if fps else None,
                "avg_latency_ms": statistics.mean(latencies) * 1000 if latencies else None,
                "p50_latency_ms": statistics.median(latencies) * 1000 if latencies else None,
                "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95) - 1] * 1000 if latencies else None,
                "fire_detection_frames": fire_detections,
                "smoke_detection_frames": smoke_detections,
                "first_detection_timestamp_seconds": first_detection,
                "candidate_alerts": alerts,
                "true_positive_events": vid_tp,
                "false_positive_events": vid_fp
            }
        )
        
    out = ROOT / "reports" / "video_evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "summary": {
            "total_videos": len(all_reports),
            "aggregate_true_positives": total_tp,
            "aggregate_false_positives": total_fp
        },
        "video_reports": all_reports
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Summary: {total_tp} TP events, {total_fp} FP events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
