# Baseline V0 Snapshot

## Git Context
- **Commit**: `b5893d7d899ef62ac8d4f6229bc9f746f0eb4371`
- **Branch**: `main`

## Model Details
- **Path**: `runs/detect/runs/yolo11n_hf_indoor_512/cpu_20e/weights/best.pt`
- **SHA-256**: `2ae9186fa8072b94ddca4f746fa4b55f777471834c5fda8da1d5bca068e7c603`
- **Size**: 5.2 MB
- **Training Config**: `configs/yolo11n_hf_indoor_512_cpu.yaml`

## Dataset V1
- **Path**: `data/processed/fire_smoke_v1`
- **Train count**: 3319
- **Validation count**: 772
- **Test count**: 824

## Validation Metrics (20 Epochs)
- **mAP50**: 0.556
- **mAP50-95**: 0.291
- **Precision**: 0.728
- **Recall**: 0.492

## Threshold Reports
- **Validation Recommended**: Fire 0.50, Smoke 0.30
- **Test Recommended**: Fire 0.35, Smoke 0.40

## CPU Benchmark
- **Average latency**: 18.35 ms
- **p50 latency**: 18.06 ms
- **p95 latency**: 20.73 ms
- **Throughput**: 54.50 FPS

## Known Limitations
- Zero dedicated negative images.
- Thresholds are tuned against positive-focused validation images, lacking true negative video testing.
