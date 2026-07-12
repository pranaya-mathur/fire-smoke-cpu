import os
import cv2
import csv
import random
from pathlib import Path
import numpy as np

def generate_synthetic_data():
    v1_images_dir = Path("data/processed/fire_smoke_v1/images/train")
    v1_labels_dir = Path("data/processed/fire_smoke_v1/labels/train")
    
    out_img_dir = Path("data/raw/synthetic_negatives/images")
    out_lbl_dir = Path("data/raw/synthetic_negatives/labels")
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    out_smoke_img_dir = Path("data/raw/synthetic_smoke/images")
    out_smoke_lbl_dir = Path("data/raw/synthetic_smoke/labels")
    out_smoke_img_dir.mkdir(parents=True, exist_ok=True)
    out_smoke_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    if not v1_images_dir.exists():
        print("V1 dataset not found.")
        return

    images = list(v1_images_dir.glob("*.jpg"))
    random.shuffle(images)
    
    neg_count = 0
    smoke_count = 0
    
    for img_path in images:
        if neg_count >= 2000 and smoke_count >= 1000:
            break
            
        lbl_path = v1_labels_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
            
        with open(lbl_path, "r") as f:
            lines = f.readlines()
            
        boxes = []
        has_fire = False
        has_smoke = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                if cls_id == 0: has_fire = True
                if cls_id == 1: has_smoke = True
                x, y, w, h = map(float, parts[1:5])
                boxes.append((x, y, w, h))
                
        if len(boxes) == 0:
            continue
            
        # Synthesize Smoke Booster
        if has_smoke and not has_fire and smoke_count < 1000:
            img = cv2.imread(str(img_path))
            if img is not None:
                # Horizontal flip for augmentation
                img_flipped = cv2.flip(img, 1)
                out_name = f"synthetic_smoke_{smoke_count}"
                cv2.imwrite(str(out_smoke_img_dir / (out_name + ".jpg")), img_flipped)
                
                with open(out_smoke_lbl_dir / (out_name + ".txt"), "w") as f:
                    for parts in [l.strip().split() for l in lines]:
                        cls_id, x, y, w, h = parts
                        # Flipped x
                        new_x = 1.0 - float(x)
                        f.write(f"{cls_id} {new_x:.6f} {y} {w} {h}\n")
                smoke_count += 1
                
        # Synthesize Negative
        if neg_count < 2000:
            img = cv2.imread(str(img_path))
            if img is not None:
                h_img, w_img = img.shape[:2]
                
                # Try to find a non-overlapping crop
                found_crop = False
                for _ in range(20): # max 20 attempts
                    crop_w, crop_h = w_img // 2, h_img // 2
                    if crop_w == 0 or crop_h == 0: break
                    
                    x_start = random.randint(0, w_img - crop_w)
                    y_start = random.randint(0, h_img - crop_h)
                    
                    crop_box = (
                        (x_start + crop_w/2) / w_img,
                        (y_start + crop_h/2) / h_img,
                        crop_w / w_img,
                        crop_h / h_img
                    )
                    
                    # Check overlap with existing boxes
                    overlap = False
                    for bx, by, bw, bh in boxes:
                        # simple center-based distance check or AABB
                        if abs(crop_box[0] - bx) < (crop_box[2] + bw)/2 and abs(crop_box[1] - by) < (crop_box[3] + bh)/2:
                            overlap = True
                            break
                    
                    if not overlap:
                        crop_img = img[y_start:y_start+crop_h, x_start:x_start+crop_w]
                        crop_img = cv2.resize(crop_img, (w_img, h_img))
                        
                        out_name = f"synthetic_negative_{neg_count}"
                        cv2.imwrite(str(out_img_dir / (out_name + ".jpg")), crop_img)
                        # Empty label file
                        open(out_lbl_dir / (out_name + ".txt"), "w").close()
                        
                        neg_count += 1
                        found_crop = True
                        break

    print(f"Synthesized {neg_count} negatives and {smoke_count} smoke boosters.")

if __name__ == "__main__":
    generate_synthetic_data()
