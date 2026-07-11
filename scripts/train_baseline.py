#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
(ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / ".cache" / "ultralytics").mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "yolo11n_512_baseline.yaml")
    args = parser.parse_args()
    import yaml
    from ultralytics import YOLO

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = YOLO(cfg["model"])
    model.train(**{k: v for k, v in cfg.items() if k != "model"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
