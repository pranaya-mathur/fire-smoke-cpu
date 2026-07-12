#!/usr/bin/env python3
import json
import yaml
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_smoke_cpu.hf_auth import get_hf_token
from fire_smoke_cpu.hf_status import HFStatus

REGISTRY_PATH = "configs/hf_dataset_registry.yaml"
OUTPUT_JSON = "reports/hf_dataset_discovery.json"
OUTPUT_MD = "reports/hf_dataset_discovery.md"

def fetch_hf_api(url, token=None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8")), None
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return None, HFStatus.AUTHENTICATION_FAILED
        elif e.code == 403:
            return None, HFStatus.GATED_ACCESS_REQUIRED
        elif e.code == 404:
            return None, HFStatus.DATASET_NOT_FOUND
        else:
            return None, HFStatus.DOWNLOAD_FAILED
    except Exception as e:
        return None, HFStatus.NETWORK_BLOCKED

def process_dataset(dataset_id, hf_id):
    metadata = {
        "dataset_id": dataset_id,
        "exists": False,
        "accessible": False,
        "public_access": False,
        "authenticated_access": False,
        "gated": False,
        "token_used": False,
        "license_acceptance_required": False,
        "network_status": "UNKNOWN",
        "download_status": "PENDING",
        "failure_category": None,
        "failure_message_sanitized": None,
        "samples": 0,
        "class_names": []
    }

    url = f"https://huggingface.co/api/datasets/{hf_id}"
    token = get_hf_token()
    
    # Try public access first
    hub_data, status = fetch_hf_api(url)
    
    if hub_data:
        metadata["exists"] = True
        metadata["accessible"] = True
        metadata["public_access"] = True
        metadata["network_status"] = "OK"
    else:
        # Try authenticated access
        if token:
            hub_data, auth_status = fetch_hf_api(url, token)
            metadata["token_used"] = True
            if hub_data:
                metadata["exists"] = True
                metadata["accessible"] = True
                metadata["authenticated_access"] = True
                metadata["network_status"] = "OK"
            else:
                metadata["failure_category"] = auth_status
                metadata["network_status"] = "FAIL"
                metadata["failure_message_sanitized"] = f"API returned {auth_status}"
                if auth_status in [HFStatus.GATED_ACCESS_REQUIRED, HFStatus.LICENSE_ACCEPTANCE_REQUIRED]:
                    metadata["gated"] = True
        else:
            metadata["failure_category"] = status
            metadata["network_status"] = "FAIL"
            metadata["failure_message_sanitized"] = f"API returned {status} (No token provided)"
            if status in [HFStatus.GATED_ACCESS_REQUIRED, HFStatus.LICENSE_ACCEPTANCE_REQUIRED]:
                metadata["gated"] = True

    if hub_data:
        tags = hub_data.get("tags", [])
        if any("license:gated" in tag or "gated" in tag for tag in tags):
            metadata["gated"] = True

    return metadata

def main():
    with open(REGISTRY_PATH, "r") as f:
        registry = yaml.safe_load(f)
    
    results = []
    for cand in registry.get("candidates", []):
        dataset_id = cand["dataset_id"]
        hf_id = cand["url"].replace("https://huggingface.co/datasets/", "")
        meta = process_dataset(dataset_id, hf_id)
        results.append(meta)
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    
    with open(OUTPUT_MD, "w") as f:
        f.write("# Hugging Face Dataset Discovery Report\n\n")
        f.write("| Dataset ID | Accessible | Public | Auth | Gated | Status |\n")
        f.write("|---|---|---|---|---|---|\n")
        for res in results:
            f.write(f"| {res['dataset_id']} | {res['accessible']} | {res['public_access']} | {res['authenticated_access']} | {res['gated']} | {res['failure_category'] or 'OK'} |\n")

if __name__ == "__main__":
    main()
