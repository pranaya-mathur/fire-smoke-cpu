# Dataset Card: SecureVU Fire/Smoke Dataset V1

## Objective

Build a leakage-aware D-Fire + MS-FSDB object detection dataset for SecureVU CCTV fire and smoke detection.

## Use Case

The target deployment is CPU inference in a surveillance system, initially with 4 RTSP cameras and one selected camera potentially running fire/smoke detection alongside event-driven face intelligence.

## Class Mapping

- `0`: fire
- `1`: smoke

## Sources

- D-Fire: https://github.com/gaia-solutions-on-demand/DFireDataset
- MS-FSDB: https://github.com/XiaoyiHan6/MS-FSDB
- MIVIA fire video dataset is external validation only.
- FASDD is Phase 2 only and not automatically mixed into V1.

## Preparation Pipeline

Run the scripts in `README.md`. Raw data stays under `data/raw`; canonical samples are materialized under `data/processed/canonical`; V1 split output is under `data/processed/fire_smoke_v1`.

## Exclusions and Duplicate Handling

All severe invalid annotations are logged in `data/manifests/invalid_annotations.csv`. Exact and near duplicates are logged in `data/manifests/exact_duplicates.csv` and `data/manifests/near_duplicates.csv`.

## Splitting

The split script uses seed `42` and inferred group IDs from source directories and filename prefixes. Exact duplicate leakage across splits is blocked.

## Current Local Statistics

As of the first local setup run, the D-Fire GitHub repository cloned successfully but the actual images/labels are hosted via official OneDrive links and still require manual download. MS-FSDB, MIVIA, and FASDD also require manual source-provider download flows. Dataset V1 has not been created yet.

- Trainable samples: `0`
- Fire-only samples: `0`
- Smoke-only samples: `0`
- Fire+smoke samples: `0`
- Negative samples: `0`

## Known Limitations

Dataset statistics will be repopulated after the required public datasets are available locally and preparation scripts pass.

## License Status

License and commercial-use status are tracked in `data/manifests/licenses.csv` and `reports/license_manifest.md`. Unknown means review required; it is not legal advice.
