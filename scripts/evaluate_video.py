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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--fire-n", type=int, default=3)
    parser.add_argument("--fire-window", type=int, default=5)
    parser.add_argument("--smoke-persistence-frames", type=int, default=10)
    args = parser.parse_args()
    import cv2
    from ultralytics import YOLO

    model = YOLO(args.model)
    all_reports = []
    for video in iter_videos(args.source):
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
            result = model.predict(frame, device=args.device, conf=args.conf, verbose=False)[0]
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
                alerts.append({"frame": frame_idx, "type": "fire_candidate"})
                fire_window.clear()
            if len(smoke_window) == smoke_window.maxlen and all(smoke_window):
                alerts.append({"frame": frame_idx, "type": "smoke_candidate"})
                smoke_window.clear()
        cap.release()
        all_reports.append(
            {
                "video": str(video),
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
            }
        )
    out = ROOT / "reports" / "video_evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_reports, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
