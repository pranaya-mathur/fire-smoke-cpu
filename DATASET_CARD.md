# Dataset Card: SecureVU Fire/Smoke Dataset V1

## Objective

Build a leakage-aware object detection dataset for SecureVU CCTV fire and smoke detection.

## Use Case

The target deployment is CPU inference in a surveillance system, initially with 4 RTSP cameras and one selected camera potentially running fire/smoke detection alongside event-driven face intelligence.

## Class Mapping

- `0`: fire
- `1`: smoke

## Sources

- FIRESENSE (priority video): Zenodo https://zenodo.org/records/836749 — fire + smoke ZIPs under `data/raw/firesense/` (temporal/eval until frame labels exist)
- D-Fire official (priority OD): https://github.com/gaia-solutions-on-demand/DFireDataset — images/labels via README Kaggle mirror under `data/raw/dfire/`
- HF Indoor Fire Smoke fallback: https://huggingface.co/datasets/KienNgyuen/Fire-Smoke-Detection (**blocked** as new booster; historical exposure)

## Preparation Pipeline

Run the scripts in `README.md`. Raw data stays under `data/raw`; canonical samples are materialized under `data/processed/canonical`; V1 split output is under `data/processed/fire_smoke_v1`.

Priority download / qualify (no training):

```bash
python scripts/download_priority_sources.py download --firesense
python scripts/qualify_priority_sources.py
```

HF Indoor preparation records the provider's original train/valid/test folders but rebuilds project splits from canonical manifests to reduce source/sequence leakage risk.

## Exclusions and Duplicate Handling

All severe invalid annotations are logged in `data/manifests/invalid_annotations.csv`. Exact and near duplicates are logged in `data/manifests/exact_duplicates.csv` and `data/manifests/near_duplicates.csv`.

## Splitting

The split script uses seed `42` and inferred group IDs from source directories and filename prefixes. Exact duplicate leakage across splits is blocked.

## Current Local Statistics

- D-Fire: prepared via `scripts/prepare_dfire.py` (~21k valid samples)
- FIRESENSE: 49 videos under `data/raw/firesense/`
- Trainable samples also populated historically by `scripts/prepare_hf_indoor.py` and `scripts/create_splits.py`

## Known Limitations

Indoor fixed-CCTV coverage remains limited; blind merges into frozen V2.1 are prohibited.

## License Status

License and commercial-use status are tracked in `data/manifests/licenses.csv` and `reports/license_manifest.md`. Unknown means review required; it is not legal advice.
