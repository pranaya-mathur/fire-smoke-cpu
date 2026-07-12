# V2.1 vs V2.2 1e Same-Split Diagnostic

{
  "v2_1_val": {
    "overall": {
      "precision": 0.5462385612824758,
      "recall": 0.43690522243713736,
      "mAP50": 0.40841908243440495,
      "mAP50_95": 0.24035147081106129
    },
    "fire": {
      "precision": 0.837963667385808,
      "recall": 0.5938104448742747,
      "mAP50": 0.6550018917079583,
      "mAP50_95": 0.4189221127984636
    },
    "smoke": {
      "precision": 0.2545134551791437,
      "recall": 0.28,
      "mAP50": 0.16183627316085158,
      "mAP50_95": 0.06178082882365886
    },
    "negatives": {
      "total_negative_images": 117,
      "images_with_false_fire": 4,
      "images_with_false_smoke": 0,
      "images_with_any_false_detection": 4,
      "false_fire_predictions": 5,
      "false_smoke_predictions": 0,
      "fp_per_image": 0.042735042735042736,
      "false_positive_image_rate": 0.03418803418803419
    },
    "checkpoint": "runs/detect/runs/detect/yolo11n_v2_1_real_512_12e/weights/best.pt",
    "checkpoint_sha256": "8eda741d3741ee8b8094ee8244d1a276f0bf7ca41d5ee3afb73099095dab6aea",
    "split": "val"
  },
  "v2_2_1e_val": {
    "overall": {
      "precision": 0.701274938991707,
      "recall": 0.4111901881207387,
      "mAP50": 0.4469507468338644,
      "mAP50_95": 0.2628416387951431
    },
    "fire": {
      "precision": 0.824072925940768,
      "recall": 0.6795232333843346,
      "mAP50": 0.7420187314963501,
      "mAP50_95": 0.46765556671848696
    },
    "smoke": {
      "precision": 0.578476952042646,
      "recall": 0.14285714285714285,
      "mAP50": 0.1518827621713787,
      "mAP50_95": 0.05802771087179931
    },
    "negatives": {
      "total_negative_images": 117,
      "images_with_false_fire": 2,
      "images_with_false_smoke": 0,
      "images_with_any_false_detection": 2,
      "false_fire_predictions": 2,
      "false_smoke_predictions": 0,
      "fp_per_image": 0.017094017094017096,
      "false_positive_image_rate": 0.017094017094017096
    },
    "checkpoint": "runs/detect/runs/detect/smoke_test_v2_2_1e/weights/best.pt",
    "checkpoint_sha256": "7229c4981bce36c867c6075200a32cc05b48a49a69d04205901190e87056099c",
    "split": "val"
  },
  "v2_1_test": {
    "overall": {
      "precision": 0.6139626994005114,
      "recall": 0.46680224323758157,
      "mAP50": 0.4688900924984477,
      "mAP50_95": 0.2212640617407668
    },
    "fire": {
      "precision": 0.6429565450099556,
      "recall": 0.509090909090909,
      "mAP50": 0.5577410179781751,
      "mAP50_95": 0.28487040836370014
    },
    "smoke": {
      "precision": 0.5849688537910671,
      "recall": 0.42451357738425416,
      "mAP50": 0.38003916701872037,
      "mAP50_95": 0.15765771511783344
    },
    "negatives": {
      "total_negative_images": 117,
      "images_with_false_fire": 9,
      "images_with_false_smoke": 1,
      "images_with_any_false_detection": 10,
      "false_fire_predictions": 10,
      "false_smoke_predictions": 1,
      "fp_per_image": 0.09401709401709402,
      "false_positive_image_rate": 0.08547008547008547
    },
    "checkpoint": "runs/detect/runs/detect/yolo11n_v2_1_real_512_12e/weights/best.pt",
    "checkpoint_sha256": "8eda741d3741ee8b8094ee8244d1a276f0bf7ca41d5ee3afb73099095dab6aea",
    "split": "test"
  },
  "v2_2_1e_test": {
    "overall": {
      "precision": 0.5574567058700708,
      "recall": 0.4771243876823025,
      "mAP50": 0.46873695774709595,
      "mAP50_95": 0.23369990344291502
    },
    "fire": {
      "precision": 0.524905516719373,
      "recall": 0.6272727272727273,
      "mAP50": 0.6018871487340932,
      "mAP50_95": 0.34145125540155385
    },
    "smoke": {
      "precision": 0.5900078950207684,
      "recall": 0.3269760480918776,
      "mAP50": 0.3355867667600987,
      "mAP50_95": 0.12594855148427625
    },
    "negatives": {
      "total_negative_images": 117,
      "images_with_false_fire": 1,
      "images_with_false_smoke": 0,
      "images_with_any_false_detection": 1,
      "false_fire_predictions": 2,
      "false_smoke_predictions": 0,
      "fp_per_image": 0.017094017094017096,
      "false_positive_image_rate": 0.008547008547008548
    },
    "checkpoint": "runs/detect/runs/detect/smoke_test_v2_2_1e/weights/best.pt",
    "checkpoint_sha256": "7229c4981bce36c867c6075200a32cc05b48a49a69d04205901190e87056099c",
    "split": "test"
  },
  "diagnosis": "SCENARIO_C_BOTH"
}
