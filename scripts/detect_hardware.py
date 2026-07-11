#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_smoke_cpu.utils import git_commit, run_command, utc_now_iso, write_json


def parse_system_profiler(text: str) -> dict:
    parsed = {}
    for raw in text.splitlines():
        line = raw.strip()
        if ": " in line:
            key, value = line.split(": ", 1)
            parsed[key.lower().replace(" ", "_").replace("(", "").replace(")", "")] = value
    return parsed


def main() -> int:
    sw_vers = run_command(["sw_vers"])
    hardware = run_command(["system_profiler", "SPHardwareDataType"], timeout=60)
    profiler = parse_system_profiler(hardware.get("stdout", ""))
    disk = shutil.disk_usage(ROOT)

    report = {
        "timestamp_utc": utc_now_iso(),
        "git_commit": git_commit(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        },
        "macos": sw_vers,
        "hardware_profiler": profiler,
        "hardware_profiler_raw_stderr": hardware.get("stderr", ""),
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "memory": {
            "total_bytes": psutil.virtual_memory().total,
            "total_gib": round(psutil.virtual_memory().total / (1024**3), 2),
            "profiler_memory": profiler.get("memory", ""),
        },
        "disk": {
            "path": str(ROOT),
            "total_gib": round(disk.total / (1024**3), 2),
            "used_gib": round(disk.used / (1024**3), 2),
            "free_gib": round(disk.free / (1024**3), 2),
        },
        "torch": {},
    }

    try:
        import torch
        import torchvision

        report["torch"] = {
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "mps_is_built": bool(torch.backends.mps.is_built()),
            "mps_is_available": bool(torch.backends.mps.is_available()),
            "mps_tensor_op_ok": False,
        }
        if torch.backends.mps.is_available():
            x = torch.ones((2, 2), device="mps")
            y = (x @ x).detach().cpu()
            report["torch"]["mps_tensor_op_ok"] = y.shape == (2, 2) and float(y[0, 0]) == 2.0
    except Exception as exc:
        report["torch_error"] = repr(exc)

    write_json(ROOT / "reports" / "hardware_report.json", report)
    md = [
        "# Hardware Report",
        "",
        f"- Timestamp UTC: `{report['timestamp_utc']}`",
        f"- macOS: `{profiler.get('os_loader_version', 'see sw_vers')}` / `{sw_vers.get('stdout', '').replace(chr(10), '; ')}`",
        f"- Model: `{profiler.get('model_name', 'unknown')}` `{profiler.get('model_identifier', '')}`",
        f"- Chip: `{profiler.get('chip', 'unknown')}`",
        f"- CPU cores: `{profiler.get('total_number_of_cores', report['cpu'])}`",
        f"- Memory: `{profiler.get('memory', str(report['memory']['total_gib']) + ' GiB')}`",
        f"- Architecture: `{platform.machine()}`",
        f"- Disk free: `{report['disk']['free_gib']} GiB` at `{ROOT}`",
        f"- Python: `{platform.python_version()}` at `{sys.executable}`",
        f"- PyTorch: `{report.get('torch', {}).get('torch_version', 'not installed')}`",
        f"- Torchvision: `{report.get('torch', {}).get('torchvision_version', 'not installed')}`",
        f"- MPS built: `{report.get('torch', {}).get('mps_is_built', 'unknown')}`",
        f"- MPS available: `{report.get('torch', {}).get('mps_is_available', 'unknown')}`",
        f"- MPS tensor op ok: `{report.get('torch', {}).get('mps_tensor_op_ok', 'unknown')}`",
        "",
        "Probe stderr is retained in `hardware_report.json` when macOS blocks a low-level field.",
    ]
    (ROOT / "reports" / "hardware_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
