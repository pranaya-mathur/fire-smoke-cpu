# fire-smoke-cpu

CPU-first SecureVU fire and smoke object-detection training repository.

## Quick Start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/detect_hardware.py
python scripts/check_environment.py
python scripts/download_datasets.py --dataset all
```

Manual downloads may be required when an official provider has browser, access, CAPTCHA, or license controls. Place archives or extracted data under the printed `data/raw/<dataset>` destination and rerun the relevant preparation script.

## Dataset Pipeline

```bash
python scripts/prepare_hf_indoor.py
python scripts/prepare_dfire.py
python scripts/prepare_ms_fsdb.py
python scripts/validate_annotations.py
python scripts/deduplicate.py
python scripts/create_splits.py
python scripts/visualize_samples.py
python scripts/train_smoke_test.py
```

Full baseline training:

```bash
python scripts/train_baseline.py --config configs/yolo11n_512_baseline.yaml
```

CPU benchmarking:

```bash
python scripts/benchmark_cpu.py --model runs/yolo11n_512_baseline/baseline/weights/best.pt
```
