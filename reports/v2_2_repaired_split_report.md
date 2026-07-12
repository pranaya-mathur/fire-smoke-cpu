# V2.2 Repaired Split Report

{
  "train_count": 6186,
  "val_count": 1757,
  "test_count": 1488,
  "fire_only_by_split": {
    "train": 2050,
    "val": 491,
    "test": 380
  },
  "smoke_only_by_split": {
    "train": 1631,
    "val": 875,
    "test": 537
  },
  "fire_smoke_by_split": {
    "train": 361,
    "val": 30,
    "test": 137
  },
  "negative_by_split": {
    "train": 2144,
    "val": 361,
    "test": 434
  },
  "source_counts_by_split": {
    "train": {
      "hf_kien_indoor_existing": 2650,
      "LibreYOLO/smoke-uvylj": 392,
      "medyoussef/fire-smoke-hardnegatives-int8": 3144
    },
    "val": {
      "hf_kien_indoor_existing": 1115,
      "LibreYOLO/smoke-uvylj": 281,
      "medyoussef/fire-smoke-hardnegatives-int8": 361
    },
    "test": {
      "hf_kien_indoor_existing": 986,
      "LibreYOLO/smoke-uvylj": 68,
      "medyoussef/fire-smoke-hardnegatives-int8": 434
    }
  },
  "fire_box_count_by_split": {
    "train": 3300,
    "val": 672,
    "test": 681
  },
  "smoke_box_count_by_split": {
    "train": 2439,
    "val": 996,
    "test": 799
  },
  "object_size_by_split": {
    "train": {
      "none": 5186,
      "tiny": 864,
      "small": 136
    },
    "val": {
      "none": 1757
    },
    "test": {
      "none": 1488
    }
  },
  "exact_cross_split_duplicates": 0,
  "near_duplicate_components_spanning_splits": 0,
  "gates": {
    "VALIDATION_HAS_FIRE_ONLY": true,
    "VALIDATION_HAS_SMOKE_ONLY": true,
    "VALIDATION_HAS_FIRE_SMOKE": true,
    "VALIDATION_HAS_NEGATIVES": true,
    "TEST_HAS_FIRE_ONLY": true,
    "TEST_HAS_SMOKE_ONLY": true,
    "TEST_HAS_FIRE_SMOKE": true,
    "TEST_HAS_NEGATIVES": true,
    "SPLIT_CLASS_BALANCE_ACCEPTABLE": true
  }
}
