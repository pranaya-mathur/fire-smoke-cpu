# Smoke Test Report

One-epoch YOLO11n smoke training completed successfully on CPU after HF Indoor passed local dataset gates.

- Command: `./.venv/bin/python scripts/train_smoke_test.py --device cpu --imgsz 512`
- Device: `cpu`
- Hardware reported by Ultralytics: `Apple M5`
- Model: `yolo11n.pt`
- Dataset YAML: `data/processed/fire_smoke_v1/fire_smoke.yaml`
- Train images scanned by Ultralytics: `3319`
- Val images scanned by Ultralytics: `772`
- Corrupt train/val images: `0`
- Run directory: `runs/smoke_test/batch_8`
- Best weights: `runs/smoke_test/batch_8/weights/best.pt`
- Last weights: `runs/smoke_test/batch_8/weights/last.pt`

## One-Epoch Metrics

- Precision: `0.46379`
- Recall: `0.20216`
- mAP50: `0.2378`
- mAP50-95: `0.11112`
- Train box loss: `2.39256`
- Train cls loss: `3.26079`
- Train dfl loss: `2.09386`
- Val box loss: `2.00821`
- Val cls loss: `2.85752`
- Val dfl loss: `1.80047`
- Approximate epoch time: `299.066` seconds
- Inference speed from final validation: `83.7ms` per image on CPU

This is a pipeline smoke test only, not a production-quality model. Smoke-class recall is especially weak after one epoch and needs full training plus false-alarm tuning before any product claim.
