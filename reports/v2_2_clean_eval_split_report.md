# V2.2 Clean Eval Split Report

{
  "train_count": 7390,
  "val_count": 1030,
  "test_count": 1011,
  "fire_only_by_split": {
    "train": 2134,
    "val": 394,
    "test": 393
  },
  "smoke_only_by_split": {
    "train": 2247,
    "val": 408,
    "test": 388
  },
  "fire_smoke_by_split": {
    "train": 466,
    "val": 30,
    "test": 32
  },
  "negative_by_split": {
    "train": 2543,
    "val": 198,
    "test": 198
  },
  "source_counts_by_split": {
    "train": {
      "hf_kien_indoor_existing": 3182,
      "LibreYOLO/smoke-uvylj": 665,
      "medyoussef/fire-smoke-hardnegatives-int8": 3543
    },
    "val": {
      "hf_kien_indoor_existing": 794,
      "LibreYOLO/smoke-uvylj": 38,
      "medyoussef/fire-smoke-hardnegatives-int8": 198
    },
    "test": {
      "hf_kien_indoor_existing": 775,
      "LibreYOLO/smoke-uvylj": 38,
      "medyoussef/fire-smoke-hardnegatives-int8": 198
    }
  },
  "fire_box_count_by_split": {
    "train": 3556,
    "val": 543,
    "test": 554
  },
  "smoke_box_count_by_split": {
    "train": 3163,
    "val": 538,
    "test": 533
  },
  "object_size_by_split": {
    "train": {
      "none": 6390,
      "tiny": 864,
      "small": 136
    },
    "val": {
      "none": 1030
    },
    "test": {
      "none": 1011
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
    "SPLIT_CLASS_BALANCE_ACCEPTABLE": true,
    "INSUFFICIENT_CLEAN_EVALUATION_SAMPLES": false
  }
}
