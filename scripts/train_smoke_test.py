#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
(ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / ".cache" / "ultralytics").mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "fire_smoke_v1" / "fire_smoke.yaml")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "smoke_test")
    args = parser.parse_args()
    if not args.data.exists():
        print(f"Dataset YAML missing: {args.data}")
        return 2
    import torch
    from ultralytics import YOLO

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running pretrained inference and 1-epoch smoke test on device={device}, imgsz={args.imgsz}, cache=False")
    model = YOLO(args.model)
    model.predict(source=str(ROOT / "data" / "processed" / "fire_smoke_v1" / "images" / "val"), imgsz=args.imgsz, device=device, max_det=20, save=False, verbose=False)
    last_error = None
    for batch in [8, 4, 2, 1]:
        try:
            result = model.train(data=str(args.data), epochs=1, imgsz=args.imgsz, batch=batch, device=device, cache=False, seed=42, project=str(args.project), name=f"batch_{batch}", workers=2)
            print(result)
            return 0
        except RuntimeError as exc:
            last_error = exc
            message = str(exc).lower()
            print(f"batch={batch} failed: {exc}")
            if "out of memory" not in message and "mps" not in message:
                break
    print(f"Smoke test failed after retries: {last_error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
