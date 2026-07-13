# D-Fire local source note

## Images + labels (present)

Official D-Fire README Kaggle mirror:

- https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo
- Local: `kaggle_smoke_fire_detection_yolo/`
- Also linked from `presplit/kaggle_splits` (train/val/test YOLO layout)
- Counts: 21,527 images + matching YOLO labels
- Prepare result: 21,187 valid canonical samples (`scripts/prepare_dfire.py`)
- Class remap: Kaggle `names: [smoke, fire]` → SecureVU `0=fire, 1=smoke`
