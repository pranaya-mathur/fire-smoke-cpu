#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
(ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / ".cache" / "ultralytics").mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "fire_smoke_v1" / "fire_smoke.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf-values", default="0.20,0.25,0.30,0.40,0.50,0.60,0.70")
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.model)
    reports = []
    for conf in [float(v) for v in args.conf_values.split(",")]:
        metrics = model.val(data=str(args.data), split="val", device=args.device, conf=conf, save_json=False, plots=True)
        reports.append({"conf": conf, "results": metrics.results_dict})
    out = ROOT / "reports" / "evaluation_thresholds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
