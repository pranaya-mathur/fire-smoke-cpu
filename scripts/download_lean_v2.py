#!/usr/bin/env python3
import json
import os
import subprocess
from pathlib import Path
import zipfile

def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def download_medyoussef():
    print("--- Downloading Medyoussef Hard Negatives ---")
    dest_dir = Path("data/raw/medyoussef")
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "fire_smoke_hardnegatives_complete.zip"
    
    if not zip_path.exists():
        url = "https://huggingface.co/datasets/medyoussef/fire-smoke-hardnegatives-int8/resolve/main/fire_smoke_hardnegatives_complete.zip"
        run_cmd(["curl", "-k", "-L", "-s", "-o", str(zip_path), url])
        
    print("Extracting Medyoussef...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)
        
def download_libreyolo():
    print("--- Downloading LibreYOLO Smoke Uvylj ---")
    dest_dir = Path("data/raw/libreyolo")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    meta_path = Path("data/raw/hf_metadata/LibreYOLO_smoke-uvylj_api.json")
    if not meta_path.exists():
        print("Metadata not found, skipping LibreYOLO.")
        return
        
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    siblings = meta.get("siblings", [])
    for sibling in siblings:
        rfilename = sibling.get("rfilename", "")
        if rfilename.endswith(".jpg") or rfilename.endswith(".txt") or rfilename.endswith("yaml"):
            out_file = dest_dir / rfilename
            if out_file.exists():
                continue
            out_file.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/LibreYOLO/smoke-uvylj/resolve/main/{rfilename}"
            subprocess.run(["curl", "-k", "-L", "-s", "-o", str(out_file), url])
            
def main():
    download_medyoussef()
    download_libreyolo()
    print("Done downloading lean datasets.")

if __name__ == "__main__":
    main()
