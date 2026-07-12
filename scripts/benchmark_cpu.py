#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time
import random
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
(ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / ".cache" / "ultralytics").mkdir(parents=True, exist_ok=True)

def get_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "N/A"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "all_samples.csv")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    import psutil
    from ultralytics import YOLO
    import torch

    # Select samples with fixed seed
    random.seed(args.seed)
    
    if args.manifest.exists():
        rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8")))
        # Filter for test split if available, otherwise just use all
        test_rows = [r for r in rows if r.get("split") == "test"]
        if test_rows:
            rows = test_rows
        random.shuffle(rows)
        sources = [row["canonical_image_path"] for row in rows[: max(1, min(len(rows), args.iterations))]]
    else:
        sources = []
        
    if not sources:
        print("Warning: No samples found. Benchmark will not run properly without images.")
        return 1

    model = YOLO(args.model)
    proc = psutil.Process()
    
    print(f"Warming up ({args.warmup} iterations)...")
    for source in sources[: args.warmup]:
        if os.path.exists(source):
            model.predict(source, imgsz=args.imgsz, device="cpu", verbose=False)
            
    print(f"Benchmarking ({args.iterations} iterations)...")
    latencies = []
    rss = []
    cpu_utils = []
    
    # reset CPU util
    psutil.cpu_percent()
    
    for i in range(args.iterations):
        source = sources[i % len(sources)]
        if not os.path.exists(source):
            continue
            
        start = time.perf_counter()
        model.predict(source, imgsz=args.imgsz, device="cpu", verbose=False)
        latencies.append(time.perf_counter() - start)
        rss.append(proc.memory_info().rss)
        cpu_utils.append(psutil.cpu_percent())
        
    if not latencies:
        print("No valid benchmark iterations completed.")
        return 1

    report = {
        "hardware": platform.processor() or platform.machine(),
        "os": platform.system() + " " + platform.release(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "model": args.model,
        "model_sha256": get_sha256(Path(args.model)),
        "model_size_mb": round(Path(args.model).stat().st_size / (1024 * 1024), 2) if Path(args.model).exists() else None,
        "imgsz": args.imgsz,
        "iterations": args.iterations,
        "batch_size": 1,
        "warmup": args.warmup,
        "avg_latency_ms": statistics.mean(latencies) * 1000,
        "p50_latency_ms": statistics.median(latencies) * 1000,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95) - 1] * 1000,
        "throughput_fps": 1 / statistics.mean(latencies) if statistics.mean(latencies) > 0 else 0,
        "peak_process_ram_mb": max(rss) / (1024 * 1024),
        "avg_process_ram_mb": statistics.mean(rss) / (1024 * 1024),
        "avg_cpu_utilization": statistics.mean(cpu_utils)
    }
    
    out = ROOT / "reports" / "cpu_benchmark.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    
    md = ["# CPU Benchmark", "", *[f"- **{k}**: `{v}`" for k, v in report.items()]]
    (ROOT / "reports" / "cpu_benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
