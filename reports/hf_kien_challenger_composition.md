# HF Kien Fallback Challenger Composition

{
  "boxes_by_split": {
    "test": {
      "fire": 554,
      "smoke": 533
    },
    "train": {
      "fire": 4605,
      "smoke": 4363
    },
    "val": {
      "fire": 543,
      "smoke": 538
    }
  },
  "dataset": "data/processed/fire_smoke_hf_kien_challenger",
  "fallback_train_cap": 1600,
  "images_by_source": {
    "LibreYOLO/smoke-uvylj": {
      "test": 38,
      "train": 665,
      "val": 38
    },
    "hf_kien_indoor_existing": {
      "test": 775,
      "train": 3182,
      "val": 794
    },
    "hf_kien_indoor_fire_smoke": {
      "train": 1600
    },
    "medyoussef/fire-smoke-hardnegatives-int8": {
      "test": 198,
      "train": 3543,
      "val": 198
    }
  },
  "images_by_split": {
    "test": 1011,
    "train": 8990,
    "val": 1030
  },
  "manifest": "data/manifests/hf_kien_challenger_all_samples.csv",
  "timestamp_utc": "2026-07-12T15:17:18.345317+00:00",
  "total_images": 11031
}
