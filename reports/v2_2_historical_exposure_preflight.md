# V2.2 Historical Exposure Preflight

{
  "git": {
    "head": "8fd8a8ba94e8d255a57a9242a0fee4c27a76d238",
    "branch": "main",
    "status_short": [
      "M scripts/v2_2_workflow.py",
      " M tests/test_v2_2_workflow.py",
      "?? reports/v2_1_historical_training_exposure.json",
      "?? reports/v2_1_historical_training_exposure.md"
    ]
  },
  "environment": {
    "python": "3.12.13",
    "torch": "2.13.0",
    "ultralytics": "8.4.92"
  },
  "checkpoint": {
    "path": "runs/detect/runs/detect/yolo11n_v2_1_real_512_12e/weights/best.pt",
    "exists": true,
    "sha256": "8eda741d3741ee8b8094ee8244d1a276f0bf7ca41d5ee3afb73099095dab6aea",
    "sha256_verified": true
  },
  "manifests": {
    "v2_1_real_all_samples": {
      "path": "data/manifests/v2_1_real_all_samples.csv",
      "exists": true,
      "rows": 8656
    },
    "v2_2_repaired_all_samples": {
      "path": "data/manifests/v2_2_repaired_all_samples.csv",
      "exists": true,
      "rows": 9431
    },
    "v2_2_repaired_near_duplicate_components": {
      "path": "data/manifests/v2_2_repaired_near_duplicate_components.csv",
      "exists": true,
      "rows": 27643
    },
    "v2_2_repaired_selected_fire_positives": {
      "path": "data/manifests/v2_2_repaired_selected_fire_positives.csv",
      "exists": true,
      "rows": 1000
    }
  }
}
