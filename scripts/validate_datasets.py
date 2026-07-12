#!/usr/bin/env python3
import os
import cv2
import json
import hashlib
import csv
from pathlib import Path

def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def validate_datasets():
    print("Validating Datasets for V2.1...")
    
    # Setup directories
    Path("data/manifests").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    
    medyoussef_base = Path("data/raw/hf_candidates/medyoussef_fire-smoke-hardnegatives-int8")
    libreyolo_base = Path("data/raw/hf_candidates/LibreYOLO_smoke-uvylj/train")
    
    corrupt_csv = open("data/manifests/v2_1_corrupt_images.csv", "w", newline="")
    corrupt_writer = csv.writer(corrupt_csv)
    corrupt_writer.writerow(["dataset_id", "image_path", "failure_reason", "file_size", "width", "height"])
    
    medyoussef_csv = open("data/manifests/medyoussef_validated_samples.csv", "w", newline="")
    med_writer = csv.writer(medyoussef_csv)
    med_writer.writerow(["sample_id", "image_path", "label_path", "width", "height", "status", "is_negative", "negative_category", "has_fire", "has_smoke", "class_ids", "annotation_valid", "license_status", "provenance_status", "included", "exclusion_reason"])
    
    libreyolo_csv = open("data/manifests/libreyolo_validated_samples.csv", "w", newline="")
    libre_writer = csv.writer(libreyolo_csv)
    libre_writer.writerow(["sample_id", "image_path", "label_path", "width", "height", "status", "is_negative", "has_fire", "has_smoke", "class_ids", "annotation_valid", "license_status", "provenance_status", "included", "exclusion_reason"])
    
    med_stats = {"total": 0, "corrupt": 0, "confirmed_negative": 0, "missing_annotation": 0}
    libre_stats = {"total": 0, "corrupt": 0, "valid_smoke": 0, "missing_annotation": 0}
    
    hashes = {}
    duplicates = []
    
    # Process Medyoussef
    if medyoussef_base.exists():
        for img_path in (medyoussef_base / "images").glob("*.jpg"):
            med_stats["total"] += 1
            lbl_path = medyoussef_base / "labels" / f"{img_path.stem}.txt"
            
            img = cv2.imread(str(img_path))
            if img is None:
                med_stats["corrupt"] += 1
                corrupt_writer.writerow(["medyoussef", str(img_path), "unreadable", img_path.stat().st_size, 0, 0])
                continue
                
            h, w = img.shape[:2]
            
            file_hash = hash_file(img_path)
            if file_hash in hashes:
                duplicates.append((str(img_path), hashes[file_hash]))
                med_writer.writerow([img_path.stem, str(img_path), str(lbl_path), w, h, "UNKNOWN", False, "unknown", False, False, "", False, "pending", False, "exact_duplicate"])
                continue
            hashes[file_hash] = str(img_path)
            
            if not lbl_path.exists():
                med_stats["missing_annotation"] += 1
                med_writer.writerow([img_path.stem, str(img_path), "", w, h, "MISSING_ANNOTATION", False, "unknown", False, False, "", False, "pending", False, "missing_label"])
            else:
                with open(lbl_path, "r") as f:
                    content = f.read().strip()
                if not content:
                    med_stats["confirmed_negative"] += 1
                    med_writer.writerow([img_path.stem, str(img_path), str(lbl_path), w, h, "CONFIRMED_NEGATIVE", True, "normal_indoor", False, False, "", True, "pending", True, ""])
                else:
                    med_writer.writerow([img_path.stem, str(img_path), str(lbl_path), w, h, "ANNOTATED_POSITIVE", False, "unknown", True, False, "0", True, "pending", False, "has_boxes_in_negative_set"])

    # Process LibreYOLO
    if libreyolo_base.exists():
        for img_path in (libreyolo_base / "images").glob("*.jpg"):
            libre_stats["total"] += 1
            lbl_path = libreyolo_base / "labels" / f"{img_path.stem}.txt"
            
            img = cv2.imread(str(img_path))
            if img is None:
                libre_stats["corrupt"] += 1
                corrupt_writer.writerow(["libreyolo", str(img_path), "unreadable", img_path.stat().st_size, 0, 0])
                continue
                
            h, w = img.shape[:2]
            
            file_hash = hash_file(img_path)
            if file_hash in hashes:
                duplicates.append((str(img_path), hashes[file_hash]))
                libre_writer.writerow([img_path.stem, str(img_path), str(lbl_path), w, h, "UNKNOWN", False, False, False, "", False, "pending", False, "exact_duplicate"])
                continue
            hashes[file_hash] = str(img_path)
            
            if not lbl_path.exists():
                libre_stats["missing_annotation"] += 1
                libre_writer.writerow([img_path.stem, str(img_path), "", w, h, "MISSING_ANNOTATION", False, False, False, "", False, "pending", False, "missing_label"])
            else:
                libre_stats["valid_smoke"] += 1
                # Read class ID
                with open(lbl_path, "r") as f:
                    content = f.read().strip()
                c_id = content.split()[0] if content else ""
                libre_writer.writerow([img_path.stem, str(img_path), str(lbl_path), w, h, "VALID_SMOKE", False, False, True, c_id, True, "pending", True, ""])

    corrupt_csv.close()
    medyoussef_csv.close()
    libreyolo_csv.close()
    
    with open("data/manifests/v2_1_exact_duplicates.csv", "w") as f:
        f.write("file1,file2\n")
        for f1, f2 in duplicates:
            f.write(f"{f1},{f2}\n")
            
    # Write reports
    with open("reports/medyoussef_hardnegative_validation.json", "w") as f:
        json.dump(med_stats, f)
    with open("reports/medyoussef_hardnegative_validation.md", "w") as f:
        f.write("# Medyoussef Hard Negatives Validation\n\n")
        for k, v in med_stats.items(): f.write(f"- {k}: {v}\n")
        
    with open("reports/libreyolo_smoke_validation.json", "w") as f:
        json.dump(libre_stats, f)
    with open("reports/libreyolo_smoke_validation.md", "w") as f:
        f.write("# LibreYOLO Smoke Validation\n\n")
        for k, v in libre_stats.items(): f.write(f"- {k}: {v}\n")

if __name__ == "__main__":
    validate_datasets()
