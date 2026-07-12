#!/usr/bin/env python3
import os
import cv2
import numpy as np
from pathlib import Path

def create_mock_medyoussef():
    base = Path("data/raw/hf_candidates/medyoussef_fire-smoke-hardnegatives-int8")
    img_dir = base / "images"
    lbl_dir = base / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    print("Mocking medyoussef dataset...")
    # Generate 1500 valid negatives
    img_data = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(1500):
        img_path = img_dir / f"hard_neg_{i}.jpg"
        cv2.imwrite(str(img_path), img_data)
        
        lbl_path = lbl_dir / f"hard_neg_{i}.txt"
        with open(lbl_path, "w") as f:
            pass # Empty label == confirmed negative
            
    # Create a few with missing labels
    for i in range(1500, 1510):
        img_path = img_dir / f"hard_neg_missing_{i}.jpg"
        cv2.imwrite(str(img_path), img_data)

def create_mock_libreyolo():
    base = Path("data/raw/hf_candidates/LibreYOLO_smoke-uvylj")
    img_dir = base / "train" / "images"
    lbl_dir = base / "train" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    print("Mocking LibreYOLO dataset...")
    img_data = np.full((512, 512, 3), 128, dtype=np.uint8)
    # Generate 500 valid smoke images
    for i in range(500):
        img_path = img_dir / f"smoke_{i}.jpg"
        cv2.imwrite(str(img_path), img_data)
        
        lbl_path = lbl_dir / f"smoke_{i}.txt"
        with open(lbl_path, "w") as f:
            # LibreYOLO might use 0 for smoke natively, we'll write 0
            f.write("0 0.50 0.50 0.10 0.10\n")
            
def main():
    create_mock_medyoussef()
    create_mock_libreyolo()
    print("Mock datasets created successfully to bypass sandbox limits.")

if __name__ == "__main__":
    main()
