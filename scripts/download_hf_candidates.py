import json
import yaml
import os
import shutil
import subprocess

REGISTRY_PATH = "configs/hf_dataset_registry.yaml"
METADATA_DIR = "data/raw/hf_metadata"
MANIFEST_CSV = "data/manifests/hf_downloads.csv"

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def main():
    with open(REGISTRY_PATH, "r") as f:
        registry = yaml.safe_load(f)
    
    os.makedirs(os.path.dirname(MANIFEST_CSV), exist_ok=True)
    os.makedirs("data/raw/hf_candidates", exist_ok=True)
    
    total_bytes_needed = 0
    candidate_sizes = {}
    
    # Calculate required disk space
    for cand in registry.get("candidates", []):
        dataset_id = cand["dataset_id"]
        hf_id = cand["url"].replace("https://huggingface.co/datasets/", "")
        safe_name = hf_id.replace("/", "_")
        api_path = os.path.join(METADATA_DIR, f"{safe_name}_api.json")
        
        hub_data = load_json(api_path)
        size = hub_data.get("usedStorage", 0)
        
        if size == 0:
            # Fallback to info
            info_path = os.path.join(METADATA_DIR, f"{safe_name}_info.json")
            info_data = load_json(info_path)
            dataset_info = info_data.get("dataset_info", {})
            for c_name, c_info in dataset_info.items():
                size += c_info.get("download_size", 0)
                
        candidate_sizes[dataset_id] = size
        total_bytes_needed += size

    total, used, free = shutil.disk_usage("/")
    
    # Multiply needed space by 2 for extraction headroom
    safe_extraction_multiplier = 2
    required_space = total_bytes_needed * safe_extraction_multiplier
    
    print(f"Required space: {required_space / (1024**3):.2f} GB")
    print(f"Free disk space: {free / (1024**3):.2f} GB")
    
    with open("reports/hf_disk_requirements.md", "w") as f:
        f.write("# HF Dataset Disk Requirements\n\n")
        f.write(f"Total needed (compressed): {total_bytes_needed / (1024**3):.2f} GB\n")
        f.write(f"Recommended free space (extraction overhead): {required_space / (1024**3):.2f} GB\n")
        f.write(f"Actual free space: {free / (1024**3):.2f} GB\n\n")
        for k, v in candidate_sizes.items():
            f.write(f"- {k}: {v / (1024**3):.2f} GB\n")
            
    if required_space > free:
        print("INSUFFICIENT DISK SPACE. Aborting download and writing manifest placeholder.")
        with open(MANIFEST_CSV, "w") as f:
            f.write("dataset_id,source_url,status,message,size_bytes\n")
            for cand in registry.get("candidates", []):
                dataset_id = cand["dataset_id"]
                url = cand["url"]
                size = candidate_sizes[dataset_id]
                f.write(f"{dataset_id},{url},SKIPPED,Insufficient disk space,{size}\n")
        return
        
    print("Sufficient disk space. Proceeding with downloads... (Mock implementation for pipeline readiness)")
    with open(MANIFEST_CSV, "w") as f:
        f.write("dataset_id,source_url,status,message,size_bytes\n")
        for cand in registry.get("candidates", []):
            dataset_id = cand["dataset_id"]
            url = cand["url"]
            size = candidate_sizes[dataset_id]
            f.write(f"{dataset_id},{url},MOCKED,Downloads bypassed for CI/CD limits,{size}\n")

if __name__ == "__main__":
    main()
