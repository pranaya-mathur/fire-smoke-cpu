#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def build_v2():
    print("Building Synthetic Dataset V2...")
    v1_dir = Path("data/processed/fire_smoke_v1")
    v2_tmp = Path("data/processed/fire_smoke_v2.tmp")
    v2_final = Path("data/processed/fire_smoke_v2")
    
    if v2_tmp.exists():
        shutil.rmtree(v2_tmp)
        
    for split in ["train", "val", "test"]:
        os.makedirs(v2_tmp / "images" / split, exist_ok=True)
        os.makedirs(v2_tmp / "labels" / split, exist_ok=True)
        
        # Copy V1
        for img in (v1_dir / "images" / split).glob("*.jpg"):
            shutil.copy2(img, v2_tmp / "images" / split / img.name)
        for lbl in (v1_dir / "labels" / split).glob("*.txt"):
            shutil.copy2(lbl, v2_tmp / "labels" / split / lbl.name)
            
    # Add synthetic data to train
    syn_neg_img = Path("data/raw/synthetic_negatives/images")
    syn_neg_lbl = Path("data/raw/synthetic_negatives/labels")
    syn_smk_img = Path("data/raw/synthetic_smoke/images")
    syn_smk_lbl = Path("data/raw/synthetic_smoke/labels")
    
    if syn_neg_img.exists():
        for img in syn_neg_img.glob("*.jpg"):
            shutil.copy2(img, v2_tmp / "images" / "train" / img.name)
        for lbl in syn_neg_lbl.glob("*.txt"):
            shutil.copy2(lbl, v2_tmp / "labels" / "train" / lbl.name)
            
    if syn_smk_img.exists():
        for img in syn_smk_img.glob("*.jpg"):
            shutil.copy2(img, v2_tmp / "images" / "train" / img.name)
        for lbl in syn_smk_lbl.glob("*.txt"):
            shutil.copy2(lbl, v2_tmp / "labels" / "train" / lbl.name)
            
    yaml_content = f"""path: {v2_final.absolute()}
train: images/train
val: images/val
test: images/test
names:
  0: fire
  1: smoke
"""
    (v2_tmp / "fire_smoke.yaml").write_text(yaml_content)
    
    if v2_final.exists():
        shutil.rmtree(v2_final)
    v2_tmp.rename(v2_final)
    
    print(f"Dataset V2 built successfully at {v2_final}")

if __name__ == "__main__":
    build_v2()
