# Official D-Fire downloads

The D-Fire GitHub clone under `DFireDataset/` does **not** include the training images.
Do not rewrite or delete existing raw files.

## Status

| Asset | Status | Location |
|---|---|---|
| Images + YOLO labels | **Present** via official README Kaggle mirror | `kaggle_smoke_fire_detection_yolo/` |
| Pre-split train/val/test | **Present** (same package) | `presplit/kaggle_splits/` |

## Images + labels source

The official D-Fire README lists this ready-to-use mirror:

- https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo
- Local tree: `kaggle_smoke_fire_detection_yolo/`

**Class-order warning:** Kaggle `data.yaml` has `names: ['smoke', 'fire']` (0=smoke, 1=fire).  
SecureVU requires `0=fire`, `1=smoke`. `scripts/prepare_dfire.py` remaps on prepare.

## After updates

```bash
python scripts/download_priority_sources.py verify
python scripts/prepare_dfire.py
python scripts/qualify_priority_sources.py
```

License: CC0-1.0 (from official D-Fire GitHub `LICENSE`). Commercial release still requires process review.
