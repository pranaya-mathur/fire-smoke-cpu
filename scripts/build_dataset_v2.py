#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def run_step(name: str, cmd: list[str]) -> bool:
    print(f"\n--- Starting: {name} ---")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"FAILED: {name}")
        return False
    print(f"SUCCESS: {name}")
    return True

def main():
    print("Building Dataset V2 Pipeline...")
    
    steps = [
        ("Download HF Candidates", [sys.executable, "scripts/download_hf_candidates.py"]),
        # (Adapters would be run here to convert to normalized rows)
        ("Cross-dataset Deduplication", [sys.executable, "scripts/deduplicate_v2_candidates.py"]),
        # (Negative QC and Visual QC would run here)
        ("Create V2 Splits (Idempotent & Class-Aware)", [sys.executable, "scripts/create_v2_splits.py"])
    ]
    
    for name, cmd in steps:
        if not run_step(name, cmd):
            sys.exit(1)
            
    print("\nDataset V2 pipeline completed successfully.")
    
if __name__ == "__main__":
    main()
