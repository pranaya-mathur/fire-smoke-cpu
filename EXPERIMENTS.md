# Experiments

## V1 Baseline

- Model: YOLO11n pretrained weights
- Image size: 512
- Device: MPS when verified stable, otherwise CPU
- Batch fallback: 8, 4, 2, 1
- Cache: false
- Seed: 42

## YOLOX-Nano Challenger Plan

After the YOLO11n baseline is complete, evaluate YOLOX-Nano from https://github.com/Megvii-BaseDetection/YOLOX using the exact same Dataset V1 split. Plan 416 and 512 input-resolution experiments. Treat MPS training support as a practical question to verify, not an assumption. Track YOLOX code license and pretrained-weight license separately.
