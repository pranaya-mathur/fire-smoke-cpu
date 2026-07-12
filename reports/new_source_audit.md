# New Source Audit

```json
{
  "archive_path": "data/raw/hf_candidates/YingjieCheng_FireSmokeDetDatasets/datasets.zip",
  "archive_sha256": "d6b5fdf61da633b659869d919fc54419624b4fa6648e742965e00428bb2083e9",
  "canonical_class_mapping": {
    "0": "fire",
    "1": "smoke"
  },
  "canonical_root": "data/processed/canonical/yingjie_firesmoke",
  "class_map_applied": {
    "0": 0,
    "1": 1
  },
  "class_mapping_evidence": "Dataset uses YOLO class ids 0/1. Sample inspection shows class-0 boxes are smaller localized flame regions and class-1 boxes are larger plume regions co-occurring in the same frames, matching canonical 0=fire / 1=smoke.",
  "corrupt_images": 0,
  "download_timestamp_utc": "2026-07-12T16:26:40.641365+00:00",
  "empty_labels_as_confirmed_negatives": 30,
  "exact_duplicates_skipped": 1194,
  "fire_box_count": 8877,
  "image_count": 6063,
  "invalid_coordinates": 11,
  "license_metadata": {
    "hub_card_license": "apache-2.0",
    "intended_use": "R&D_and_commercial_evaluation_compatible",
    "status": "apache-2.0"
  },
  "manifest": "data/manifests/yingjie_firesmoke_samples.csv",
  "missing_labels_excluded": 2195,
  "note": "Missing labels were excluded, never treated as negatives.",
  "original_class_mapping": {
    "0": "fire",
    "1": "smoke"
  },
  "smoke_box_count": 6324,
  "source": "YingjieCheng/FireSmokeDetDatasets",
  "source_url": "https://huggingface.co/datasets/YingjieCheng/FireSmokeDetDatasets",
  "status": "PASS",
  "unsupported_classes": 0,
  "version": "hf_main_datasets.zip"
}
```
