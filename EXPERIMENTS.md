# Experiments

## V1 Baseline

- Model: YOLO11n pretrained weights
- Image size: 512
- Device: MPS when verified stable, otherwise CPU
- Batch fallback: 8, 4, 2, 1
- Cache: false
- Seed: 42

### HF Indoor Smoke Test

- Dataset source: HF Indoor fallback (`hf_kien_indoor`)
- Prepared samples: `4915`
- Split: `3319` train, `772` val, `824` test
- Leakage checks: `0` exact duplicate cross-split issues, `0` near-duplicate cross-split pairs
- One-epoch CPU smoke test: completed
- Result: mAP50 `0.2378`, mAP50-95 `0.11112`
- Report: `reports/smoke_test_report.md`

### HF Indoor Proper CPU Baseline

- Config: `configs/yolo11n_hf_indoor_512_cpu.yaml`
- Purpose: move beyond the one-epoch smoke test with a 20-epoch CPU fine-tune, then tune fire/smoke confidence thresholds from validation predictions.
- Threshold tuning script: `scripts/tune_thresholds.py`
- Status: completed 20 CPU epochs
- Best weights: `runs/detect/runs/yolo11n_hf_indoor_512/cpu_20e/weights/best.pt`
- Final validation: precision `0.728`, recall `0.492`, mAP50 `0.556`, mAP50-95 `0.291`
- Reports:
  - `reports/hf_indoor_20e_report.md`
  - `reports/threshold_tuning_val.md`
  - `reports/threshold_tuning_test.md`

## YOLOX-Nano Challenger Plan

After the YOLO11n baseline is complete, evaluate YOLOX-Nano from https://github.com/Megvii-BaseDetection/YOLOX using the exact same Dataset V1 split. Plan 416 and 512 input-resolution experiments. Treat MPS training support as a practical question to verify, not an assumption. Track YOLOX code license and pretrained-weight license separately.
