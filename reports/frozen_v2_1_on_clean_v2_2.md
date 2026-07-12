# Frozen V2.1 on Clean V2.2

{
  "metric_scope_note": "official_clean_baseline; earlier repaired V2.2 metrics are historically_exposure_unverified",
  "val": {
    "overall": {
      "precision": 0.7200352109646346,
      "recall": 0.5156691792122793,
      "mAP50": 0.5731916673672928,
      "mAP50_95": 0.3000098012399409
    },
    "fire": {
      "precision": 0.7654182110931101,
      "recall": 0.6279926335174953,
      "mAP50": 0.6999789912096326,
      "mAP50_95": 0.39380742976961847
    },
    "smoke": {
      "precision": 0.6746522108361591,
      "recall": 0.4033457249070632,
      "mAP50": 0.44640434352495306,
      "mAP50_95": 0.20621217271026343
    },
    "negatives": {
      "total_negative_images": 198,
      "images_with_false_fire": 6,
      "images_with_false_smoke": 4,
      "images_with_any_false_detection": 9,
      "false_fire_predictions": 8,
      "false_smoke_predictions": 5,
      "fp_per_image": 0.06565656565656566,
      "false_positive_image_rate": 0.045454545454545456
    },
    "checkpoint": "runs/detect/runs/detect/yolo11n_v2_1_real_512_12e/weights/best.pt",
    "checkpoint_sha256": "8eda741d3741ee8b8094ee8244d1a276f0bf7ca41d5ee3afb73099095dab6aea",
    "split": "val"
  },
  "test": {
    "overall": {
      "precision": 0.7104578350981139,
      "recall": 0.5257889068754614,
      "mAP50": 0.5865874653350596,
      "mAP50_95": 0.29338478375825716
    },
    "fire": {
      "precision": 0.7340504524298054,
      "recall": 0.5956678700361011,
      "mAP50": 0.6770544259001852,
      "mAP50_95": 0.35956848014618437
    },
    "smoke": {
      "precision": 0.6868652177664224,
      "recall": 0.45590994371482174,
      "mAP50": 0.4961205047699339,
      "mAP50_95": 0.22720108737032993
    },
    "negatives": {
      "total_negative_images": 198,
      "images_with_false_fire": 8,
      "images_with_false_smoke": 4,
      "images_with_any_false_detection": 11,
      "false_fire_predictions": 12,
      "false_smoke_predictions": 4,
      "fp_per_image": 0.08080808080808081,
      "false_positive_image_rate": 0.05555555555555555
    },
    "checkpoint": "runs/detect/runs/detect/yolo11n_v2_1_real_512_12e/weights/best.pt",
    "checkpoint_sha256": "8eda741d3741ee8b8094ee8244d1a276f0bf7ca41d5ee3afb73099095dab6aea",
    "split": "test"
  }
}
