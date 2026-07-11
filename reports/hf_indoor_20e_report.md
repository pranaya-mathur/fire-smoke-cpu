# HF Indoor 20-Epoch CPU Baseline

## Training

- Command: `./.venv/bin/python scripts/train_baseline.py --config configs/yolo11n_hf_indoor_512_cpu.yaml`
- Model: `yolo11n.pt`
- Dataset YAML: `data/processed/fire_smoke_v1/fire_smoke.yaml`
- Split: `3319` train, `772` val, `824` test
- Device: CPU
- Image size: `512`
- Epochs: `20`
- Run directory: `runs/detect/runs/yolo11n_hf_indoor_512/cpu_20e`
- Best weights: `runs/detect/runs/yolo11n_hf_indoor_512/cpu_20e/weights/best.pt`
- Model size: `5.2 MB`

## Validation Metrics

| Run | Precision | Recall | mAP50 | mAP50-95 | CPU validation inference |
|---|---:|---:|---:|---:|---:|
| 1-epoch smoke test | 0.464 | 0.202 | 0.238 | 0.111 | 83.7 ms/img |
| 20-epoch baseline | 0.728 | 0.492 | 0.556 | 0.291 | 73.5 ms/img |

The 20-epoch run is a real baseline rather than a pipeline smoke test. Aggregate recall improved by about `2.43x`, and mAP50 improved by about `2.34x`.

## Threshold Tuning

Validation recommendations:

- Fire threshold: `0.50`
- Smoke threshold: `0.30`
- Smoke at `0.30`: precision `0.505`, recall `0.389`, unmatched predictions `0.131/img`
- Fire at `0.50`: precision `0.888`, recall `0.620`, unmatched predictions `0.069/img`

Held-out test recommendations:

- Fire threshold: `0.35`
- Smoke threshold: `0.40`
- Smoke at `0.40`: precision `0.605`, recall `0.440`, unmatched predictions `0.258/img`
- Fire at `0.35`: precision `0.701`, recall `0.674`, unmatched predictions `0.153/img`

Pilot gate recommendation:

- Use class-specific thresholds rather than one global threshold.
- Start with `fire=0.40`, `smoke=0.40` for a balanced pilot setting.
- If smoke miss-rate matters more than false alarms in the pilot, test `smoke=0.30` with operator review.
- Keep fire and smoke thresholds separately configurable in deployment.

## CPU Benchmark

Standalone benchmark using `scripts/benchmark_cpu.py` over 100 single-image iterations:

- Average latency: `18.35 ms`
- p50 latency: `18.06 ms`
- p95 latency: `20.73 ms`
- Throughput: `54.50 FPS`
- Peak process RAM: `424.88 MB`
- Average process RAM: `419.25 MB`

## Caveat

The false-positive rates above are unmatched predictions on labeled validation/test images. This is useful for threshold shaping, but it is not a real false-alarm rate for deployed CCTV. Before any production claim, build a negative validation set with indoor CCTV/no-fire/no-smoke scenes and re-run threshold tuning against that negative set.
