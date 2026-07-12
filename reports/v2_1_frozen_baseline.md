# V2.1 Frozen Baseline

{
  "baseline_id": "candidate_v2_1_baseline",
  "checkpoint_path": "runs/detect/runs/detect/yolo11n_v2_1_real_512_12e/weights/best.pt",
  "checkpoint_sha256": "8eda741d3741ee8b8094ee8244d1a276f0bf7ca41d5ee3afb73099095dab6aea",
  "raw_results_csv_last_row": {
    "epoch": "6",
    "time": "4581.22",
    "train/box_loss": "1.24843",
    "train/cls_loss": "0.97672",
    "train/dfl_loss": "1.29543",
    "metrics/precision(B)": "0.7408",
    "metrics/recall(B)": "0.49537",
    "metrics/mAP50(B)": "0.55655",
    "metrics/mAP50-95(B)": "0.32204",
    "val/box_loss": "1.39007",
    "val/cls_loss": "1.54227",
    "val/dfl_loss": "1.36494",
    "lr/pg0": "0.000146875",
    "lr/pg1": "0.000146875",
    "lr/pg2": "0.000146875"
  },
  "negative_eval_metrics": {
    "v0": {
      "model_path": "runs/detect/runs/yolo11n_hf_indoor_512/cpu_20e/weights/best.pt",
      "total_negative_images": 278,
      "images_with_false_fire": 58,
      "images_with_false_smoke": 42,
      "images_with_any_false_detection": 97,
      "false_fire_count": 101,
      "false_smoke_count": 67,
      "fp_per_image": 0.60431654676259,
      "false_positive_image_rate": 0.3489208633093525,
      "elapsed_seconds": 25.21852707862854,
      "false_positives_by_negative_category": {
        "unknown": {
          "images": 278,
          "any_fp": 97,
          "false_fire_images": 58,
          "false_smoke_images": 42,
          "false_fire_count": 101,
          "false_smoke_count": 67
        }
      }
    },
    "v2": {
      "model_path": "runs/detect/runs/detect/yolo11n_v2_512_20e/weights/best.pt",
      "total_negative_images": 278,
      "images_with_false_fire": 45,
      "images_with_false_smoke": 29,
      "images_with_any_false_detection": 72,
      "false_fire_count": 76,
      "false_smoke_count": 46,
      "fp_per_image": 0.43884892086330934,
      "false_positive_image_rate": 0.2589928057553957,
      "elapsed_seconds": 23.275054931640625,
      "false_positives_by_negative_category": {
        "unknown": {
          "images": 278,
          "any_fp": 72,
          "false_fire_images": 45,
          "false_smoke_images": 29,
          "false_fire_count": 76,
          "false_smoke_count": 46
        }
      }
    },
    "real_v2_1": {
      "model_path": "runs/detect/runs/detect/yolo11n_v2_1_real_512_12e/weights/best.pt",
      "total_negative_images": 278,
      "images_with_false_fire": 12,
      "images_with_false_smoke": 1,
      "images_with_any_false_detection": 13,
      "false_fire_count": 17,
      "false_smoke_count": 1,
      "fp_per_image": 0.06474820143884892,
      "false_positive_image_rate": 0.046762589928057555,
      "elapsed_seconds": 21.612664937973022,
      "false_positives_by_negative_category": {
        "unknown": {
          "images": 278,
          "any_fp": 13,
          "false_fire_images": 12,
          "false_smoke_images": 1,
          "false_fire_count": 17,
          "false_smoke_count": 1
        }
      }
    }
  },
  "cpu_benchmark_reference": {
    "v0": {
      "model_path": "runs/detect/runs/yolo11n_hf_indoor_512/cpu_20e/weights/best.pt",
      "sample_count": 200,
      "avg_latency_ms": 38.94804225565167,
      "p50_latency_ms": 37.18489551101811,
      "p95_latency_ms": 51.48755626869388,
      "avg_rss_bytes": 435361136,
      "peak_rss_bytes": 436240384,
      "model_size_bytes": 5449562
    },
    "v2": {
      "model_path": "runs/detect/runs/detect/yolo11n_v2_512_20e/weights/best.pt",
      "sample_count": 200,
      "avg_latency_ms": 36.26460060448153,
      "p50_latency_ms": 35.579312505433336,
      "p95_latency_ms": 43.687768414383754,
      "avg_rss_bytes": 447328583,
      "peak_rss_bytes": 447791104,
      "model_size_bytes": 5447962
    },
    "real_v2_1": {
      "model_path": "runs/detect/runs/detect/yolo11n_v2_1_real_512_12e/weights/best.pt",
      "sample_count": 200,
      "avg_latency_ms": 37.19499044294935,
      "p50_latency_ms": 36.28329199273139,
      "p95_latency_ms": 44.99005482357461,
      "avg_rss_bytes": 458775347,
      "peak_rss_bytes": 458899456,
      "model_size_bytes": 5447962
    }
  },
  "model_size_mb": 5.196
}
