#!/usr/bin/env python3
from ultralytics import YOLO
from pathlib import Path

def main():
    print("Starting V2 20-Epoch Max Training...")
    model_path = "runs/detect/runs/yolo11n_hf_indoor_512/cpu_20e/weights/best.pt"
    if not Path(model_path).exists():
        model_path = "yolo11n.pt"
        
    model = YOLO(model_path)
    
    # Run the main training
    results = model.train(
        data=str(Path("data/processed/fire_smoke_v2/fire_smoke.yaml").absolute()),
        epochs=20,
        patience=5,
        imgsz=512,
        device="cpu",
        batch=8,
        workers=2,
        cache=False,
        seed=42,
        optimizer="AdamW",
        lr0=0.0003,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        hsv_h=0.01,
        hsv_s=0.30,
        hsv_v=0.30,
        degrees=0.0,
        translate=0.10,
        scale=0.30,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.30,
        close_mosaic=5,
        mixup=0.0,
        copy_paste=0.0,
        project="runs/detect",
        name="yolo11n_v2_512_20e"
    )
    
    print("V2 Training Completed.")

if __name__ == "__main__":
    main()
