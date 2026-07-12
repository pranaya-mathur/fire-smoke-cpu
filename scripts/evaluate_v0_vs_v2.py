#!/usr/bin/env python3
import json
import csv
from pathlib import Path
import os

def load_metrics(results_csv_path):
    metrics = {}
    if not os.path.exists(results_csv_path):
        return metrics
    with open(results_csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return metrics
        last_row = rows[-1]
        # YOLO11 results.csv headers:
        # epoch, train/box_loss, train/cls_loss, train/dfl_loss, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B), ...
        metrics = {k.strip(): float(v) for k, v in last_row.items() if v.strip() != ""}
    return metrics

def compare():
    print("Evaluating V0 vs V2...")
    v0_path = "runs/detect/runs/yolo11n_hf_indoor_512/cpu_20e/results.csv"
    v2_path = "runs/detect/runs/detect/yolo11n_v2_512_20e/results.csv"
    
    v0_metrics = load_metrics(v0_path)
    v2_metrics = load_metrics(v2_path)
    
    report_md = Path("reports/v0_vs_v2_comparison.md")
    report_md.parent.mkdir(parents=True, exist_ok=True)
    
    out = [
        "# V0 vs V2 Model Comparison\n",
        "## Overall Metrics (Validation Set)\n",
        "| Metric | V0 Baseline | V2 Fine-Tuned |",
        "|---|---|---|"
    ]
    
    keys_of_interest = ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"]
    for k in keys_of_interest:
        v0_val = v0_metrics.get(k, 0.0)
        v2_val = v2_metrics.get(k, 0.0)
        out.append(f"| {k} | {v0_val:.4f} | {v2_val:.4f} |")
        
    out.extend([
        "\n## File Size & Hardware\n",
        "- **V0 Model Size:** ~5.2 MB",
        "- **V2 Model Size:** ~5.2 MB (Same architecture)",
        "\n*Note: Class-specific metrics and negative false-positive rates require running the full evaluate_video.py script on test videos.*"
    ])
    
    report_md.write_text("\n".join(out))
    
    report_json = Path("reports/v0_vs_v2_comparison.json")
    json.dump({"v0": v0_metrics, "v2": v2_metrics}, report_json.open("w"), indent=2)
    
    print(f"Report generated at {report_md}")

if __name__ == "__main__":
    compare()
