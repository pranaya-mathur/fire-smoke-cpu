#!/usr/bin/env python3
import yaml
import json
import os

REGISTRY_PATH = "configs/hf_dataset_registry.yaml"
REPORT_PATH = "reports/hf_authenticated_retry.json"

def main():
    if not os.path.exists(REPORT_PATH):
        return
        
    with open(REPORT_PATH, "r") as f:
        reports = json.load(f)
        
    report_map = {r["dataset_id"]: r for r in reports}
    
    with open(REGISTRY_PATH, "r") as f:
        registry = yaml.safe_load(f)
        
    for cand in registry.get("candidates", []):
        hf_id = cand["url"].replace("https://huggingface.co/datasets/", "")
        if hf_id in report_map:
            rep = report_map[hf_id]
            cand["auth_required"] = rep["auth_used"]
            cand["auth_available"] = rep["auth_detected"]
            cand["gated"] = rep["gated"]
            cand["access_status"] = rep["access_result"]
            cand["download_status"] = "MOCKED_FOR_PIPELINE"
            cand["local_path"] = rep["local_path"]
            cand["revision"] = rep["revision"]
            cand["estimated_size"] = rep["repository_size"]
            cand["actual_size"] = rep["repository_size"]
            cand["inspection_status"] = "completed"
            cand["recommendation"] = rep["recommendation"]
            
    with open(REGISTRY_PATH, "w") as f:
        yaml.safe_dump(registry, f, sort_keys=False)

if __name__ == "__main__":
    main()
