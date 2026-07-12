#!/usr/bin/env python3
import json
import sys
from ultralytics import YOLO
from pathlib import Path

MAX_EPOCHS = 12
QUALITY_GATE = Path("reports/v2_1_real_quality_gate.json")
DATA_YAML = Path("data/processed/fire_smoke_v2_1_real/fire_smoke.yaml")


def require_real_quality_gate():
    if not QUALITY_GATE.exists():
        raise SystemExit("Refusing to train: missing reports/v2_1_real_quality_gate.json")
    report = json.loads(QUALITY_GATE.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise SystemExit(f"Refusing to train: real V2.1 quality gate is {report.get('status')}")
    gates = report.get("gates", {})
    if gates.get("NO_MOCK_DATA_FOR_REAL_TRAINING") is not True:
        raise SystemExit("Refusing to train: NO_MOCK_DATA_FOR_REAL_TRAINING did not pass")
    if not DATA_YAML.exists():
        raise SystemExit(f"Refusing to train: missing real dataset YAML at {DATA_YAML}")


def main():
    print("Starting real V2.1 training with a 12-epoch hard ceiling...")
    require_real_quality_gate()
    model_path = "runs/detect/runs/detect/yolo11n_v2_512_20e/weights/best.pt"
    if not Path(model_path).exists():
        raise SystemExit(f"Refusing to train: missing V2 checkpoint {model_path}")
        
    model = YOLO(model_path)
    
    # Run the main training
    results = model.train(
        data=str(DATA_YAML.absolute()),
        epochs=MAX_EPOCHS,
        patience=3,
        imgsz=512,
        device="cpu",
        batch=8,
        workers=2,
        cache=False,
        seed=42,
        optimizer="AdamW",
        lr0=0.00025,
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
        close_mosaic=3,
        mixup=0.0,
        copy_paste=0.0,
        project="runs/detect",
        name="yolo11n_v2_1_real_512_12e"
    )
    
    print("Real V2.1 Training Completed.")

if __name__ == "__main__":
    main()
