# V2.2 Model Artifact Manifest

{
  "model_id": "securevu_fire_smoke_v2_2",
  "checkpoint_path": "/Users/mobcoderid-296/Desktop/fire-smoke/fire-smoke-cpu/runs/detect/runs/detect/smoke_test_v2_2_1e/weights/best.pt",
  "checkpoint_sha256": "7229c4981bce36c867c6075200a32cc05b48a49a69d04205901190e87056099c",
  "parent_checkpoint_path": "runs/detect/runs/detect/yolo11n_v2_1_real_512_12e/weights/best.pt",
  "parent_checkpoint_sha256": "8eda741d3741ee8b8094ee8244d1a276f0bf7ca41d5ee3afb73099095dab6aea",
  "git_commit": "16b5ec9e9a16f779e9335e2f0ca8ddc91f9320de",
  "dataset_version": "fire_smoke_v2_2",
  "dataset_manifest_path": "data/manifests/v2_2_all_samples.csv",
  "dataset_manifest_sha256": "2cc67f397fc6039d568b209920fd3cdad385d7f651bf7839ea1ca30b9637b0bc",
  "training_config": {
    "imgsz": 512,
    "epochs_requested": 8,
    "patience": 2,
    "batch": 8,
    "workers": 2,
    "optimizer": "AdamW",
    "lr0": 0.0002,
    "mosaic": 0.25,
    "parent": "runs/detect/runs/detect/yolo11n_v2_1_real_512_12e/weights/best.pt"
  },
  "config_sha256": "1e56d205985e7998739776be3482e6820e94ad2899bbe1d0c4a2347d3f64c85a",
  "epochs_requested": 8,
  "epochs_completed": 1,
  "artifact_status": "SMOKE_TEST_ONLY",
  "artifact_location": "/Users/mobcoderid-296/Desktop/fire-smoke/fire-smoke-cpu/runs/detect/runs/detect/smoke_test_v2_2_1e/weights"
}
