# Download Attempt Report

Timestamp UTC: `2026-07-13T00:00:00Z`

## D-Fire (images + labels)

- Official GitHub repository is cloned at `data/raw/dfire/DFireDataset` (code/docs only; no training images in the clone).
- Training package: official README Kaggle mirror `sayedgamal99/smoke-fire-detection-yolo`.
- Local tree: `data/raw/dfire/kaggle_smoke_fire_detection_yolo/` (+ `presplit/`).
- Status: `PRESENT`.
- Class-order note: Kaggle YAML is `0=smoke, 1=fire`; SecureVU remaps via `scripts/prepare_dfire.py` to `0=fire, 1=smoke`.

## FIRESENSE

- Zenodo record `836749`.
- Local tree: `data/raw/firesense/`.
- Status: `PRESENT` (videos for temporal validation).
