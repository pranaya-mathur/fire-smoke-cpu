import json
import yaml
import os
import re

REGISTRY_PATH = "configs/hf_dataset_registry.yaml"
OUTPUT_JSON = "reports/hf_dataset_discovery.json"
OUTPUT_MD = "reports/hf_dataset_discovery.md"

def extract_json_from_md(md_path):
    if not os.path.exists(md_path):
        return {}
    with open(md_path, "r") as f:
        content = f.read()
    
    # JSON usually starts after `---`
    parts = content.split("---")
    if len(parts) >= 2:
        json_str = parts[-1].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    return {}

def process_dataset(dataset_id, hf_id, api_md, info_md):
    metadata = {
        "dataset_id": dataset_id,
        "exists": False,
        "repo_metadata": None,
        "readme": False,
        "license_tag": "unknown",
        "has_license_file": False,
        "configs": 0,
        "splits": 0,
        "samples": 0,
        "columns": [],
        "image_storage": "unknown",
        "has_bounding_boxes": False,
        "annotation_format": "unknown",
        "class_names": [],
        "class_ids": [],
        "has_fire": False,
        "has_smoke": False,
        "has_negative": False,
        "distractors": False,
        "video_frames": False,
        "grouping_inferable": False,
        "synthetic": False,
        "cctv_relevant": False,
        "provenance_documented": False,
    }

    hub_data = extract_json_from_md(api_md)
    if hub_data and "id" in hub_data:
        metadata["exists"] = True
        # don't store full repo_metadata in JSON to save space, but extract info
        
        siblings = [s.get("rfilename", "") for s in hub_data.get("siblings", [])]
        metadata["readme"] = "README.md" in siblings
        metadata["has_license_file"] = any(s.lower() in ["license", "license.txt", "license.md"] for s in siblings)

        tags = hub_data.get("tags", [])
        for tag in tags:
            if tag.startswith("license:"):
                metadata["license_tag"] = tag.split("license:")[1]
        
        desc = hub_data.get("description", "").lower()
        if "yolo" in desc or "yolo" in " ".join(tags).lower() or hf_id.lower().find("yolo") != -1:
            metadata["annotation_format"] = "yolo"
            metadata["has_bounding_boxes"] = True
        elif "coco" in desc or "coco" in " ".join(tags).lower():
            metadata["annotation_format"] = "coco"
            metadata["has_bounding_boxes"] = True
        elif "voc" in desc:
            metadata["annotation_format"] = "voc"
            metadata["has_bounding_boxes"] = True
            
        if "fire" in desc or "fire" in hf_id.lower():
            metadata["has_fire"] = True
        if "smoke" in desc or "smoke" in hf_id.lower():
            metadata["has_smoke"] = True
        if "synthetic" in desc:
            metadata["synthetic"] = True
        if "cctv" in desc or "surveillance" in desc:
            metadata["cctv_relevant"] = True

    info_data = extract_json_from_md(info_md)
    if info_data:
        dataset_info = info_data.get("dataset_info", {})
        metadata["configs"] = len(dataset_info)
        
        for config_name, config_info in dataset_info.items():
            splits = config_info.get("splits", {})
            metadata["splits"] += len(splits)
            for split_name, split_info in splits.items():
                metadata["samples"] += split_info.get("num_examples", 0)
            
            features = config_info.get("features", {})
            metadata["columns"] = list(features.keys())
            
            if "image" in features:
                metadata["image_storage"] = "embedded_parquet"
            if "objects" in features or "bboxes" in features or "bbox" in features:
                metadata["has_bounding_boxes"] = True
                
            for k, v in features.items():
                if isinstance(v, dict) and "feature" in v:
                    v = v["feature"]
                if isinstance(v, dict) and v.get("_type") == "ClassLabel":
                    metadata["class_names"].extend(v.get("names", []))
                if k == "objects" and isinstance(v, dict) and "feature" in v:
                    obj_features = v["feature"]
                    if "category" in obj_features:
                        cat_info = obj_features["category"]
                        if isinstance(cat_info, dict) and cat_info.get("_type") == "ClassLabel":
                            metadata["class_names"].extend(cat_info.get("names", []))
    
    metadata["class_names"] = list(set(metadata["class_names"]))
    metadata["class_ids"] = list(range(len(metadata["class_names"])))
    
    if any("fire" in c.lower() for c in metadata["class_names"]):
        metadata["has_fire"] = True
    if any("smoke" in c.lower() for c in metadata["class_names"]):
        metadata["has_smoke"] = True

    return metadata

def main():
    with open(REGISTRY_PATH, "r") as f:
        registry = yaml.safe_load(f)
    
    mappings = {
        "medyoussef/fire-smoke-hardnegatives-int8": (152, 153),
        "LibreYOLO/smoke-uvylj": (154, 155),
        "YingjieCheng/FireSmokeDetDatasets": (156, 157),
        "KienNgyuen/Fire-Smoke-Detection": (158, 159),
        "hiennguyen9874/fire-smoke-detection": (160, 161),
        "betasecond/jimei-fire-smoke-yolo-dataset": (162, 163),
    }

    results = []
    for cand in registry.get("candidates", []):
        dataset_id = cand["dataset_id"]
        hf_id = cand["url"].replace("https://huggingface.co/datasets/", "")
        
        api_step, info_step = mappings.get(hf_id, (0,0))
        base_dir = "/Users/mobcoderid-296/.gemini/antigravity-ide/brain/1b882836-e1bf-4709-84df-be949ac6830a/.system_generated/steps"
        api_md = f"{base_dir}/{api_step}/content.md"
        info_md = f"{base_dir}/{info_step}/content.md"
        
        meta = process_dataset(dataset_id, hf_id, api_md, info_md)
        results.append(meta)
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    
    with open(OUTPUT_MD, "w") as f:
        f.write("# Hugging Face Dataset Discovery Report\n\n")
        f.write("## Summary\n\n")
        f.write("| Dataset ID | Status | License | Samples | Classes |\n")
        f.write("|---|---|---|---|---|\n")
        for res in results:
            status = "OK" if res["exists"] else "NOT FOUND"
            classes = ", ".join(res["class_names"]) if res["class_names"] else "Unknown"
            f.write(f"| {res['dataset_id']} | {status} | {res['license_tag']} | {res['samples']} | {classes} |\n")
        
        f.write("\n## Details\n\n")
        for res in results:
            f.write(f"### {res['dataset_id']}\n")
            f.write(f"- **Exists**: {res['exists']}\n")
            f.write(f"- **License Tag**: {res['license_tag']}\n")
            f.write(f"- **Has License File**: {res['has_license_file']}\n")
            f.write(f"- **Samples**: {res['samples']}\n")
            f.write(f"- **Has Bounding Boxes**: {res['has_bounding_boxes']}\n")
            f.write(f"- **Annotation Format**: {res['annotation_format']}\n")
            f.write(f"- **Classes**: {res['class_names']}\n")
            f.write("\n")

if __name__ == "__main__":
    main()
