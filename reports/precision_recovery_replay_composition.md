# Precision Recovery Replay Composition

```json
{
  "class_counts": {
    "fire": 165,
    "negative": 358,
    "smoke": 356
  },
  "dataset_yaml": "data/processed/fire_smoke_precision_recovery/fire_smoke.yaml",
  "deviations": [],
  "error_category_counts": {
    "CONFUSING_HARDNEG_SOURCE": 55,
    "FALSE_FIRE_ON_NEGATIVE": 27,
    "FALSE_SMOKE_ON_NEGATIVE": 9,
    "FIRE_MISSED": 50,
    "NEAR_BOUNDARY_FALSE_FIRE": 29,
    "NEAR_BOUNDARY_FALSE_SMOKE": 20,
    "SMOKE_MISSED": 100,
    "unknown": 218
  },
  "no_clean_val_test_in_train": true,
  "percentages": {
    "hard_fire": 5.69,
    "hard_negatives": 15.93,
    "hard_smoke": 11.38,
    "replay": 67.01
  },
  "replay_ratio": 0.6700796359499431,
  "role_counts": {
    "new_hard_fire": 50,
    "new_hard_negative": 140,
    "new_hard_smoke": 100,
    "replay": 589
  },
  "status": "PASS",
  "targets": {
    "hard_fire_pct": "5-7%",
    "hard_negatives_pct": "12-18%",
    "hard_smoke_pct": "10-12%",
    "replay_pct": "65-70%"
  },
  "total_train": 879
}
```
