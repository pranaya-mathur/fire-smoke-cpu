import csv
import json
from pathlib import Path

def analyze():
    print("Analyzing negative candidates for V2...")
    out_md = Path("reports/negative_quality_report.md")
    out_md.parent.mkdir(exist_ok=True, parents=True)
    
    with out_md.open("w") as f:
        f.write("# Negative Quality Report\n\n")
        f.write("## Overview\n")
        f.write("Analyzes the 'hard negatives' (e.g. from medyoussef dataset).\n\n")
        f.write("## Distribution\n")
        f.write("- Total Negatives Analyzed: 0 (Mocked)\n")
        f.write("- Detected Easy Negatives: 0\n")
        f.write("- Detected Hard Negatives: 0\n")
        
    print(f"Report written to {out_md}")

if __name__ == "__main__":
    analyze()
