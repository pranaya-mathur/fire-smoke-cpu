#!/usr/bin/env python3
import json
from ultralytics import YOLO
from pathlib import Path

QUALITY_GATE = Path("reports/v2_1_real_quality_gate.json")
DATA_YAML = Path("data/processed/fire_smoke_v2_1_real/fire_smoke.yaml")


def require_real_quality_gate():
    if not QUALITY_GATE.exists():
        raise SystemExit("Refusing smoke test: missing real V2.1 quality gate")
    report = json.loads(QUALITY_GATE.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise SystemExit(f"Refusing smoke test: real V2.1 quality gate is {report.get('status')}")
    if report.get("gates", {}).get("NO_MOCK_DATA_FOR_REAL_TRAINING") is not True:
        raise SystemExit("Refusing smoke test: NO_MOCK_DATA_FOR_REAL_TRAINING did not pass")
    if not DATA_YAML.exists():
        raise SystemExit(f"Refusing smoke test: missing {DATA_YAML}")


def main():
    print("Running real V2.1 Smoke Test (exactly 1 epoch)...")
    require_real_quality_gate()
    model_path = "runs/detect/runs/detect/yolo11n_v2_512_20e/weights/best.pt"
    
    if not Path(model_path).exists():
        raise SystemExit(f"Refusing smoke test: missing V2 checkpoint {model_path}")
        
    model = YOLO(model_path)
    
    results = model.train(
        data=str(DATA_YAML.absolute()),
        epochs=1,
        imgsz=512,
        device="cpu",
        batch=8,
        workers=2,
        cache=False,
        seed=42,
        project="runs/smoke_test_v2_1_real",
        name="batch_8"
    )
    
    print("Real V2.1 smoke test complete.")
    
    # Generate report
    report_path = Path("reports/v2_1_real_smoke_test.md")
    report_path.parent.mkdir(exist_ok=True, parents=True)
    report_path.write_text("# Real Dataset V2.1 Smoke Test Report\n\n- Requested epochs: 1\n- Device: CPU\n- Data YAML: data/processed/fire_smoke_v2_1_real/fire_smoke.yaml\n- Completed without script-level errors.\n")
    Path("reports/v2_1_real_smoke_test.json").write_text(json.dumps({"requested_epochs": 1, "device": "cpu", "status": "PASS"}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
