#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
(ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / ".cache" / "ultralytics").mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "all_samples.csv")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    import psutil
    from ultralytics import YOLO

    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8")))
    sources = [row["canonical_image_path"] for row in rows[: max(1, min(len(rows), args.iterations))]]
    model = YOLO(args.model)
    proc = psutil.Process()
    for source in sources[: args.warmup]:
        model.predict(source, device="cpu", verbose=False)
    latencies = []
    rss = []
    for i in range(args.iterations):
        source = sources[i % len(sources)]
        start = time.perf_counter()
        model.predict(source, device="cpu", verbose=False)
        latencies.append(time.perf_counter() - start)
        rss.append(proc.memory_info().rss)
    report = {
        "model": args.model,
        "model_size_mb": round(Path(args.model).stat().st_size / (1024 * 1024), 2) if Path(args.model).exists() else None,
        "iterations": args.iterations,
        "batch_size": 1,
        "avg_latency_ms": statistics.mean(latencies) * 1000,
        "p50_latency_ms": statistics.median(latencies) * 1000,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95) - 1] * 1000,
        "throughput_fps": 1 / statistics.mean(latencies),
        "peak_process_ram_mb": max(rss) / (1024 * 1024),
        "avg_process_ram_mb": statistics.mean(rss) / (1024 * 1024),
    }
    out = ROOT / "reports" / "cpu_benchmark.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = ["# CPU Benchmark", "", *[f"- {k}: `{v}`" for k, v in report.items()]]
    (ROOT / "reports" / "cpu_benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
