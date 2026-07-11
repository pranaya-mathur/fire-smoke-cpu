#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
(ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / ".cache" / "ultralytics").mkdir(parents=True, exist_ok=True)

from fire_smoke_cpu.utils import utc_now_iso, write_json


PACKAGES = [
    "torch",
    "torchvision",
    "ultralytics",
    "cv2",
    "PIL",
    "numpy",
    "pandas",
    "yaml",
    "tqdm",
    "imagehash",
    "sklearn",
    "matplotlib",
    "psutil",
    "lxml",
    "gdown",
    "pytest",
]


def main() -> int:
    report = {"timestamp_utc": utc_now_iso(), "python": sys.version, "packages": {}, "mps": {}}
    ok = True
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            report["packages"][name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            ok = False
            report["packages"][name] = {"error": repr(exc)}
    try:
        import torch

        report["mps"]["is_built"] = bool(torch.backends.mps.is_built())
        report["mps"]["is_available"] = bool(torch.backends.mps.is_available())
        report["mps"]["tensor_op_ok"] = False
        if torch.backends.mps.is_available():
            a = torch.arange(9, dtype=torch.float32, device="mps").reshape(3, 3)
            b = torch.eye(3, dtype=torch.float32, device="mps")
            c = (a @ b).cpu()
            report["mps"]["tensor_op_ok"] = float(c[2, 2]) == 8.0
    except Exception as exc:
        ok = False
        report["mps"]["error"] = repr(exc)
    write_json(ROOT / "reports" / "environment_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
