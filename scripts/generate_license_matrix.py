import json
import csv
import os

DISCOVERY_JSON = "reports/hf_dataset_discovery.json"
OUTPUT_MD = "reports/hf_license_provenance_matrix.md"
OUTPUT_CSV = "reports/hf_license_provenance_matrix.csv"

def generate():
    if not os.path.exists(DISCOVERY_JSON):
        print("Discovery JSON not found.")
        return
        
    with open(DISCOVERY_JSON, "r") as f:
        datasets = json.load(f)
        
    records = []
    for ds in datasets:
        ds_id = ds["dataset_id"]
        license_tag = ds.get("license_tag", "unknown")
        has_file = ds.get("has_license_file", False)
        
        # Heuristics for provenance/commercial use
        commercial = "unknown"
        provenance = "unknown"
        
        if "apache" in license_tag.lower() or "cc-by" in license_tag.lower() or "mit" in license_tag.lower():
            commercial = "confirmed"
        if license_tag == "unknown":
            commercial = "requires review"
            
        rec = {
            "dataset_id": ds_id,
            "hub_license_tag": license_tag,
            "readme_statement": "Requires human review",
            "actual_license_file": has_file,
            "source_attribution": "Requires human review",
            "upstream_url": "Requires human review",
            "commercial_use_status": commercial,
            "provenance_status": provenance,
            "recommendation": "Review" if commercial != "confirmed" else "Proceed"
        }
        records.append(rec)
        
    with open(OUTPUT_CSV, "w", newline="") as f:
        fields = list(records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        
    with open(OUTPUT_MD, "w") as f:
        f.write("# Hugging Face License & Provenance Matrix\n\n")
        f.write("| Dataset | Hub Tag | License File | Commercial Use | Recommendation |\n")
        f.write("|---|---|---|---|---|\n")
        for r in records:
            f.write(f"| {r['dataset_id']} | {r['hub_license_tag']} | {r['actual_license_file']} | {r['commercial_use_status']} | {r['recommendation']} |\n")

if __name__ == "__main__":
    generate()
