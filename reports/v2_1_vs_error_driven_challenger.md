# V2.1 vs Error-Driven Challenger

```json
{
  "deltas": {
    "test": {
      "fire": {
        "mAP50": {
          "abs": -0.002944920501961601,
          "baseline": 0.6770544259001852,
          "challenger": 0.6741095053982236,
          "rel": -0.004349606751401336
        },
        "mAP50_95": {
          "abs": 0.005211333598213408,
          "baseline": 0.35956848014618437,
          "challenger": 0.3647798137443978,
          "rel": 0.014493299290568279
        },
        "precision": {
          "abs": -0.009313479199215213,
          "baseline": 0.7340504524298054,
          "challenger": 0.7247369732305902,
          "rel": -0.012687791647544591
        },
        "recall": {
          "abs": -0.0018050541516245744,
          "baseline": 0.5956678700361011,
          "challenger": 0.5938628158844765,
          "rel": -0.0030303030303030732
        }
      },
      "negatives": {
        "false_positive_image_rate": {
          "abs": 0.015151515151515152,
          "baseline": 0.05555555555555555,
          "challenger": 0.0707070707070707
        }
      },
      "overall": {
        "mAP50": {
          "abs": -0.0008934425503916987,
          "baseline": 0.5865874653350596,
          "challenger": 0.5856940227846679,
          "rel": -0.001523119062698285
        },
        "mAP50_95": {
          "abs": 0.0018145670966013827,
          "baseline": 0.29338478375825716,
          "challenger": 0.29519935085485854,
          "rel": 0.006184939359692722
        },
        "precision": {
          "abs": -0.02484104788438879,
          "baseline": 0.7104578350981139,
          "challenger": 0.6856167872137251,
          "rel": -0.034964844720106796
        },
        "recall": {
          "abs": 0.0009736455320675796,
          "baseline": 0.5257889068754614,
          "challenger": 0.526762552407529,
          "rel": 0.001851780285463873
        }
      },
      "smoke": {
        "mAP50": {
          "abs": 0.00115803540117837,
          "baseline": 0.4961205047699339,
          "challenger": 0.4972785401711123,
          "rel": 0.002334181695867189
        },
        "mAP50_95": {
          "abs": -0.0015821994050105592,
          "baseline": 0.22720108737032993,
          "challenger": 0.22561888796531937,
          "rel": -0.006963872503090748
        },
        "precision": {
          "abs": -0.040368616569562366,
          "baseline": 0.6868652177664224,
          "challenger": 0.64649660119686,
          "rel": -0.05877225331169739
        },
        "recall": {
          "abs": 0.0037523452157598447,
          "baseline": 0.45590994371482174,
          "challenger": 0.4596622889305816,
          "rel": 0.008230452674897108
        }
      }
    },
    "val": {
      "fire": {
        "mAP50": {
          "abs": -0.00505729470116989,
          "baseline": 0.6999789912096326,
          "challenger": 0.6949216965084627,
          "rel": -0.007224923554391807
        },
        "mAP50_95": {
          "abs": -0.0005876700929785539,
          "baseline": 0.39380742976961847,
          "challenger": 0.3932197596766399,
          "rel": -0.001492277820462522
        },
        "precision": {
          "abs": -0.04170736639411721,
          "baseline": 0.7654182110931101,
          "challenger": 0.7237108446989929,
          "rel": -0.05448964473232747
        },
        "recall": {
          "abs": 0.02209944751381221,
          "baseline": 0.6279926335174953,
          "challenger": 0.6500920810313076,
          "rel": 0.03519061583577721
        }
      },
      "negatives": {
        "false_positive_image_rate": {
          "abs": 0.020202020202020207,
          "baseline": 0.045454545454545456,
          "challenger": 0.06565656565656566
        }
      },
      "overall": {
        "mAP50": {
          "abs": -0.012707974170135805,
          "baseline": 0.5731916673672928,
          "challenger": 0.560483693197157,
          "rel": -0.022170549387963666
        },
        "mAP50_95": {
          "abs": -0.004626321997930671,
          "baseline": 0.3000098012399409,
          "challenger": 0.29538347924201025,
          "rel": -0.0154205695240958
        },
        "precision": {
          "abs": -0.05524470400764825,
          "baseline": 0.7200352109646346,
          "challenger": 0.6647905069569864,
          "rel": -0.07672500339759308
        },
        "recall": {
          "abs": 0.017555299965084536,
          "baseline": 0.5156691792122793,
          "challenger": 0.5332244791773638,
          "rel": 0.03404372545960859
        }
      },
      "smoke": {
        "mAP50": {
          "abs": -0.020358653639101776,
          "baseline": 0.44640434352495306,
          "challenger": 0.4260456898858513,
          "rel": -0.045605859204557155
        },
        "mAP50_95": {
          "abs": -0.008664973902882789,
          "baseline": 0.20621217271026343,
          "challenger": 0.19754719880738064,
          "rel": -0.04201970130569078
        },
        "precision": {
          "abs": -0.0687820416211794,
          "baseline": 0.6746522108361591,
          "challenger": 0.6058701692149797,
          "rel": -0.1019518509187592
        },
        "recall": {
          "abs": 0.013011152416356864,
          "baseline": 0.4033457249070632,
          "challenger": 0.4163568773234201,
          "rel": 0.032258064516129
        }
      }
    }
  },
  "error_driven_challenger_1e": {
    "test": {
      "checkpoint": "runs/detect/runs/detect/v2_1_error_driven_replay_challenger_1e/weights/best.pt",
      "checkpoint_sha256": "c7e80922d3e39d5b82c3f1fb13e35a8914789db6d16cdafdaa34d17472972311",
      "fire": {
        "mAP50": 0.6741095053982236,
        "mAP50_95": 0.3647798137443978,
        "precision": 0.7247369732305902,
        "recall": 0.5938628158844765
      },
      "negatives": {
        "false_fire_predictions": 15,
        "false_positive_image_rate": 0.0707070707070707,
        "false_smoke_predictions": 7,
        "fp_per_image": 0.1111111111111111,
        "images_with_any_false_detection": 14,
        "images_with_false_fire": 8,
        "images_with_false_smoke": 7,
        "total_negative_images": 198
      },
      "overall": {
        "mAP50": 0.5856940227846679,
        "mAP50_95": 0.29519935085485854,
        "precision": 0.6856167872137251,
        "recall": 0.526762552407529
      },
      "smoke": {
        "mAP50": 0.4972785401711123,
        "mAP50_95": 0.22561888796531937,
        "precision": 0.64649660119686,
        "recall": 0.4596622889305816
      },
      "split": "test"
    },
    "val": {
      "checkpoint": "runs/detect/runs/detect/v2_1_error_driven_replay_challenger_1e/weights/best.pt",
      "checkpoint_sha256": "c7e80922d3e39d5b82c3f1fb13e35a8914789db6d16cdafdaa34d17472972311",
      "fire": {
        "mAP50": 0.6949216965084627,
        "mAP50_95": 0.3932197596766399,
        "precision": 0.7237108446989929,
        "recall": 0.6500920810313076
      },
      "negatives": {
        "false_fire_predictions": 9,
        "false_positive_image_rate": 0.06565656565656566,
        "false_smoke_predictions": 7,
        "fp_per_image": 0.08080808080808081,
        "images_with_any_false_detection": 13,
        "images_with_false_fire": 8,
        "images_with_false_smoke": 6,
        "total_negative_images": 198
      },
      "overall": {
        "mAP50": 0.560483693197157,
        "mAP50_95": 0.29538347924201025,
        "precision": 0.6647905069569864,
        "recall": 0.5332244791773638
      },
      "smoke": {
        "mAP50": 0.4260456898858513,
        "mAP50_95": 0.19754719880738064,
        "precision": 0.6058701692149797,
        "recall": 0.4163568773234201
      },
      "split": "val"
    }
  },
  "eval_dataset": "data/processed/fire_smoke_v2_2_clean_eval",
  "eval_manifest": "data/manifests/v2_2_clean_eval_all_samples.csv",
  "frozen_v2_1": {
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
  }
}
```
