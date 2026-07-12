# Error-Driven Replay Composition

```json
{
  "class_counts": {
    "fire": 206,
    "negative": 241,
    "smoke": 411
  },
  "dataset_yaml": "data/processed/fire_smoke_error_driven_replay/fire_smoke.yaml",
  "deviations": [
    "hard_neg_available_13_lt_desired_50"
  ],
  "error_category_counts": {
    "FALSE_FIRE_ON_NEGATIVE": 8,
    "FALSE_SMOKE_ON_NEGATIVE": 5,
    "FIRE_MISSED": 90,
    "SMOKE_MISSED": 180
  },
  "fire_smoke_ratio": {
    "fire": 206,
    "smoke": 411
  },
  "new_hard_fire": 90,
  "new_hard_negative": 13,
  "new_hard_smoke": 180,
  "no_clean_val_test_in_train": true,
  "object_size_buckets": {
    "large": 127,
    "medium": 93,
    "none": 588,
    "small": 40,
    "tiny": 10
  },
  "replay_ratio": 0.6701631701631702,
  "role_counts": {
    "new_hard_fire": 90,
    "new_hard_negative": 13,
    "new_hard_smoke": 180,
    "replay": 575
  },
  "status": "PASS",
  "targets": {
    "hard_negatives_pct": "~10%",
    "new_hard_fire_pct": "5-10%",
    "new_hard_smoke_pct": "15-20%",
    "replay_pct": "65-70%"
  },
  "total_train": 858
}
```
