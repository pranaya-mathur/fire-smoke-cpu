#!/usr/bin/env python3
import json
from pathlib import Path
import os

def check_download(dataset_name, expected_path):
    path = Path(expected_path)
    if not path.exists():
        return {"status": "NOT_DOWNLOADED", "files": 0, "images": 0, "size": 0}
        
    files = list(path.rglob("*"))
    images = [f for f in files if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    
    if len(images) == 0:
        return {"status": "EMPTY_DIRECTORY", "files": len(files), "images": 0, "size": total_size}
        
    return {"status": "DOWNLOAD_COMPLETE", "files": len(files), "images": len(images), "size": total_size}

def main():
    datasets = {
        "medyoussef": "data/raw/hf_candidates/medyoussef_fire-smoke-hardnegatives-int8",
        "LibreYOLO": "data/raw/hf_candidates/LibreYOLO_smoke-uvylj",
        "D-Fire": "data/raw/dfire/kaggle_smoke_fire_detection_yolo",
        "FIRESENSE": "data/raw/firesense",
    }
    
    results = {}
    for name, p in datasets.items():
        results[name] = check_download(name, p)
        
    Path("reports").mkdir(exist_ok=True)
    with open("reports/hf_download_integrity.json", "w") as f:
        json.dump(results, f, indent=2)
        
    with open("reports/hf_download_integrity.md", "w") as f:
        f.write("# HF Download Integrity Report\n\n")
        f.write("| Dataset | Status | Files | Images | Size (Bytes) |\n")
        f.write("|---|---|---|---|---|\n")
        for name, data in results.items():
            f.write(f"| {name} | {data['status']} | {data['files']} | {data['images']} | {data['size']} |\n")

if __name__ == "__main__":
    main()
