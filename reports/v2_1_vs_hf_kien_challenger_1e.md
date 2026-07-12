# V2.1 vs HF Kien Fallback Challenger 1e

{
  "deltas": {
    "test_fire_mAP50": {
      "absolute": -0.04048723687742806,
      "relative": -0.059799087530663146
    },
    "test_fire_mAP50_95": {
      "absolute": -0.023236745139905368,
      "relative": -0.06462397685820057
    },
    "test_fire_precision": {
      "absolute": 0.06213977508739843,
      "relative": 0.08465327537324914
    },
    "test_fire_recall": {
      "absolute": -0.08090696748513737,
      "relative": -0.13582563632353364
    },
    "test_overall_mAP50": {
      "absolute": -0.06631944084690056,
      "relative": -0.11305976476844558
    },
    "test_overall_mAP50_95": {
      "absolute": -0.029916871410312407,
      "relative": -0.10197144864528242
    },
    "test_overall_precision": {
      "absolute": -0.027982746836792805,
      "relative": -0.03938692129833208
    },
    "test_overall_recall": {
      "absolute": -0.06296755503712781,
      "relative": -0.11975824178436371
    },
    "test_smoke_mAP50": {
      "absolute": -0.092151644816373,
      "relative": -0.18574447927546656
    },
    "test_smoke_mAP50_95": {
      "absolute": -0.036596997680719445,
      "relative": -0.16107756395138903
    },
    "test_smoke_precision": {
      "absolute": -0.11810526876098415,
      "relative": -0.1719482450211177
    },
    "test_smoke_recall": {
      "absolute": -0.04502814258911819,
      "relative": -0.09876543209876543
    },
    "val_fire_mAP50": {
      "absolute": -0.04491321327666775,
      "relative": -0.06416365896789746
    },
    "val_fire_mAP50_95": {
      "absolute": -0.02050556883159299,
      "relative": -0.05207004053628182
    },
    "val_fire_precision": {
      "absolute": 0.06857898093683223,
      "relative": 0.08959674586118499
    },
    "val_fire_recall": {
      "absolute": -0.07366482504604044,
      "relative": -0.11730205278592364
    },
    "val_overall_mAP50": {
      "absolute": -0.05402769425347187,
      "relative": -0.09425764073233067
    },
    "val_overall_mAP50_95": {
      "absolute": -0.029366104906696944,
      "relative": -0.09788381841302114
    },
    "val_overall_precision": {
      "absolute": 0.001352844548684784,
      "relative": 0.0018788588781267679
    },
    "val_overall_recall": {
      "absolute": -0.04054988464197934,
      "relative": -0.07863546296081167
    },
    "val_smoke_mAP50": {
      "absolute": -0.06314217523027621,
      "relative": -0.1414461488696216
    },
    "val_smoke_mAP50_95": {
      "absolute": -0.03822664098180109,
      "relative": -0.1853752883711239
    },
    "val_smoke_precision": {
      "absolute": -0.06587329183946256,
      "relative": -0.09764037052190146
    },
    "val_smoke_recall": {
      "absolute": -0.0074349442379182396,
      "relative": -0.018433179723502363
    }
  },
  "frozen_v2_1": {
    "metric_scope_note": "official_clean_baseline; earlier repaired V2.2 metrics are historically_exposure_unverified",
    "test": {
      "checkpoint": "runs/detect/runs/detect/yolo11n_v2_1_real_512_12e/weights/best.pt",
      "checkpoint_sha256": "8eda741d3741ee8b8094ee8244d1a276f0bf7ca41d5ee3afb73099095dab6aea",
      "fire": {
        "mAP50": 0.6770544259001852,
        "mAP50_95": 0.35956848014618437,
        "precision": 0.7340504524298054,
        "recall": 0.5956678700361011
      },
      "negatives": {
        "false_fire_predictions": 12,
        "false_positive_image_rate": 0.05555555555555555,
        "false_smoke_predictions": 4,
        "fp_per_image": 0.08080808080808081,
        "images_with_any_false_detection": 11,
        "images_with_false_fire": 8,
        "images_with_false_smoke": 4,
        "total_negative_images": 198
      },
      "overall": {
        "mAP50": 0.5865874653350596,
        "mAP50_95": 0.29338478375825716,
        "precision": 0.7104578350981139,
        "recall": 0.5257889068754614
      },
      "smoke": {
        "mAP50": 0.4961205047699339,
        "mAP50_95": 0.22720108737032993,
        "precision": 0.6868652177664224,
        "recall": 0.45590994371482174
      },
      "split": "test"
    },
    "val": {
      "checkpoint": "runs/detect/runs/detect/yolo11n_v2_1_real_512_12e/weights/best.pt",
      "checkpoint_sha256": "8eda741d3741ee8b8094ee8244d1a276f0bf7ca41d5ee3afb73099095dab6aea",
      "fire": {
        "mAP50": 0.6999789912096326,
        "mAP50_95": 0.39380742976961847,
        "precision": 0.7654182110931101,
        "recall": 0.6279926335174953
      },
      "negatives": {
        "false_fire_predictions": 8,
        "false_positive_image_rate": 0.045454545454545456,
        "false_smoke_predictions": 5,
        "fp_per_image": 0.06565656565656566,
        "images_with_any_false_detection": 9,
        "images_with_false_fire": 6,
        "images_with_false_smoke": 4,
        "total_negative_images": 198
      },
      "overall": {
        "mAP50": 0.5731916673672928,
        "mAP50_95": 0.3000098012399409,
        "precision": 0.7200352109646346,
        "recall": 0.5156691792122793
      },
      "smoke": {
        "mAP50": 0.44640434352495306,
        "mAP50_95": 0.20621217271026343,
        "precision": 0.6746522108361591,
        "recall": 0.4033457249070632
      },
      "split": "val"
    }
  },
  "hf_kien_challenger_1e": {
    "test": {
      "checkpoint": "runs/detect/runs/detect/hf_kien_v2_1_challenger_1e/weights/best.pt",
      "checkpoint_sha256": "9c33dc538366a453fed12052a8f5bc5a8f16c68debc6b448224aef6da252f19f",
      "fire": {
        "mAP50": 0.6365671890227571,
        "mAP50_95": 0.336331735006279,
        "precision": 0.7961902275172038,
        "recall": 0.5147609025509637
      },
      "negatives": {
        "false_fire_predictions": 4,
        "false_positive_image_rate": 0.025252525252525252,
        "false_smoke_predictions": 1,
        "fp_per_image": 0.025252525252525252,
        "images_with_any_false_detection": 5,
        "images_with_false_fire": 4,
        "images_with_false_smoke": 1,
        "total_negative_images": 198
      },
      "overall": {
        "mAP50": 0.520268024488159,
        "mAP50_95": 0.26346791234794475,
        "precision": 0.6824750882613211,
        "recall": 0.4628213518383336
      },
      "smoke": {
        "mAP50": 0.4039688599535609,
        "mAP50_95": 0.19060408968961048,
        "precision": 0.5687599490054382,
        "recall": 0.41088180112570355
      },
      "split": "test"
    },
    "val": {
      "checkpoint": "runs/detect/runs/detect/hf_kien_v2_1_challenger_1e/weights/best.pt",
      "checkpoint_sha256": "9c33dc538366a453fed12052a8f5bc5a8f16c68debc6b448224aef6da252f19f",
      "fire": {
        "mAP50": 0.6550657779329648,
        "mAP50_95": 0.3733018609380255,
        "precision": 0.8339971920299424,
        "recall": 0.5543278084714549
      },
      "negatives": {
        "false_fire_predictions": 5,
        "false_positive_image_rate": 0.050505050505050504,
        "false_smoke_predictions": 6,
        "fp_per_image": 0.05555555555555555,
        "images_with_any_false_detection": 10,
        "images_with_false_fire": 5,
        "images_with_false_smoke": 6,
        "total_negative_images": 198
      },
      "overall": {
        "mAP50": 0.5191639731138209,
        "mAP50_95": 0.270643696333244,
        "precision": 0.7213880555133194,
        "recall": 0.47511929457029994
      },
      "smoke": {
        "mAP50": 0.38326216829467685,
        "mAP50_95": 0.16798553172846234,
        "precision": 0.6087789189966966,
        "recall": 0.395910780669145
      },
      "split": "val"
    }
  }
}
