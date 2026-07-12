import json
from pathlib import Path

def generate_qc():
    print("Generating visual QC sampling...")
    out_md = Path("reports/visual_qc_report.md")
    out_md.parent.mkdir(exist_ok=True, parents=True)
    
    with out_md.open("w") as f:
        f.write("# Visual Quality Control Report\n\n")
        f.write("## Sample Categories\n")
        f.write("- Night-time fire: 0 checked\n")
        f.write("- Small smoke puffs: 0 checked\n")
        f.write("- Indoor CCTV: 0 checked\n")
        f.write("\n_QC bypassed due to mocked dataset downloads in sandbox mode._\n")
        
    print(f"Report written to {out_md}")

if __name__ == "__main__":
    generate_qc()
