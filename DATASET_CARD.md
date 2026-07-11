# Dataset Card: SecureVU Fire/Smoke Dataset V1

## Objective

Build a leakage-aware object detection dataset for SecureVU CCTV fire and smoke detection. The original V1 target remains D-Fire + MS-FSDB, but the active local fallback source is HF Indoor because the official D-Fire/MS-FSDB data archives are blocked behind provider-controlled download flows.

## Use Case

The target deployment is CPU inference in a surveillance system, initially with 4 RTSP cameras and one selected camera potentially running fire/smoke detection alongside event-driven face intelligence.

## Class Mapping

- `0`: fire
- `1`: smoke

## Sources

- HF Indoor Fire Smoke fallback: https://huggingface.co/datasets/KienNgyuen/Fire-Smoke-Detection, archive `Indoor Fire Smoke.zip`
- D-Fire: https://github.com/gaia-solutions-on-demand/DFireDataset
- MS-FSDB: https://github.com/XiaoyiHan6/MS-FSDB
- MIVIA fire video dataset is external validation only.
- FASDD is Phase 2 only and not automatically mixed into V1.

## Preparation Pipeline

Run the scripts in `README.md`. Raw data stays under `data/raw`; canonical samples are materialized under `data/processed/canonical`; V1 split output is under `data/processed/fire_smoke_v1`.

HF Indoor preparation records the provider's original train/valid/test folders but rebuilds project splits from canonical manifests to reduce source/sequence leakage risk.

## Exclusions and Duplicate Handling

All severe invalid annotations are logged in `data/manifests/invalid_annotations.csv`. Exact and near duplicates are logged in `data/manifests/exact_duplicates.csv` and `data/manifests/near_duplicates.csv`.

## Splitting

The split script uses seed `42` and inferred group IDs from source directories and filename prefixes. Exact duplicate leakage across splits is blocked.

## Current Local Statistics

The D-Fire GitHub repository cloned successfully but the actual images/labels are hosted via official OneDrive links and still require manual download. MS-FSDB and MIVIA also require provider-controlled download flows. FASDD was discovered but not downloaded because it is Phase 2 and larger than available free disk.

HF Indoor is now the local fallback V1 source pending continued license/source review and deduplication checks.

- Trainable samples: populated by `scripts/prepare_hf_indoor.py` and `scripts/create_splits.py`
- Fire-only samples: populated by manifest reports
- Smoke-only samples: populated by manifest reports
- Fire+smoke samples: populated by manifest reports
- Negative samples: populated by manifest reports

## Known Limitations

Dataset statistics will be repopulated after the required public datasets are available locally and preparation scripts pass.

## License Status

License and commercial-use status are tracked in `data/manifests/licenses.csv` and `reports/license_manifest.md`. Unknown means review required; it is not legal advice.
