# Alternative Dataset QC Report

These datasets were downloaded as alternatives only and are not merged into Dataset V1.

- Archive: `data/raw/hf_kien_fire_smoke/CCTV Smoke & Fire Emergency Detection Dataset.zip`
  - size bytes: `339006279`
  - sha256: `7b1a8cbebe6c71d64a7510b88bcfedc7610a0a5e3495704e5307fd4a9eae69d5`
- Archive: `data/raw/hf_kien_fire_smoke/Indoor Fire Smoke.zip`
  - size bytes: `200547120`
  - sha256: `f5e4b452bc0e74508366b4ce3f62d7479d1a790a7438a1f0ebee145a216d7b6a`

## hf_kien_cctv
- Images: `240`
- Missing labels: `1`
- Empty labels: `1`
- Bad images: `0`
- Box counts by raw class id: `{'0': 121, '1': 141}`
- Image counts by split: `{'all': 240}`
- Visual QC: class `0` appears to be fire/flame and class `1` appears to be smoke.
- Action: keep excluded from training until the license conflict, `nc` mismatch, and missing label are resolved.

## hf_kien_indoor
- Images: `5000`
- Missing labels: `0`
- Empty labels: `0`
- Bad images: `0`
- Box counts by raw class id: `{'0': 3592, '1': 3343}`
- Image counts by split: `{'test': 750, 'train': 3500, 'valid': 750}`
- Visual QC: class `0` appears to be fire/flame and class `1` appears to be smoke.
- Action: confirm source terms and then run group-aware deduplication before using it for training; visible near-duplicate/frame-sequence patterns are present.

## License / Mapping Warnings

- `hf_kien_cctv`: do not use for commercial training yet. HF metadata conflict observed: page tag shows `cc-by-nc-4.0`, while card text says `CC BY 4.0`; archive `data.yaml` says `nc: 1` but defines classes `0: fire` and `1: smoke`.
- `hf_kien_indoor`: Roboflow YAML says `CC BY 4.0`, but class names are numeric only (`0`, `1`); semantic class mapping must be verified before conversion to canonical `0=fire, 1=smoke`.
- Neither dataset is wired into Dataset V1 or any training manifest.

Visual contact sheets are in this folder.
