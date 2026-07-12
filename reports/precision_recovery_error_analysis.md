# Precision Recovery Error Analysis

```json
{
  "analysis_rows": 872,
  "category_counts": {
    "CONFUSING_HARDNEG_SOURCE": 800,
    "FALSE_FIRE_ON_NEGATIVE": 27,
    "FALSE_SMOKE_ON_NEGATIVE": 9,
    "NEAR_BOUNDARY_FALSE_FIRE": 35,
    "NEAR_BOUNDARY_FALSE_SMOKE": 21,
    "QUESTIONABLE_POSITIVE_EXCLUDED": 50
  },
  "confusing_hardneg_source": 800,
  "direct_false_fire": 27,
  "direct_false_smoke": 9,
  "near_boundary_negatives": 56,
  "questionable_examples": [
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_test_fire000757"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_test_fire000808"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_test_fire000816"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_test_fire000821"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_test_fire000822"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_test_fire000824"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_test_fire000825"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_test_fire000839"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002187"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002204"
    },
    {
      "class": "smoke",
      "reason": "dense_noisy_annotation",
      "sample_id": "yingjie_train_fire002206"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002210"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002215"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002221"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002230"
    },
    {
      "class": "smoke",
      "reason": "dense_noisy_annotation",
      "sample_id": "yingjie_train_fire002236"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002241"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002253"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002279"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002287"
    },
    {
      "class": "smoke",
      "reason": "dense_noisy_annotation",
      "sample_id": "yingjie_train_fire002313"
    },
    {
      "class": "smoke",
      "reason": "dense_noisy_annotation",
      "sample_id": "yingjie_train_fire002327"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002328"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002329"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002380"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002391"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002397"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002401"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002405"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002407"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002411"
    },
    {
      "class": "smoke",
      "reason": "dense_noisy_annotation",
      "sample_id": "yingjie_train_fire002433"
    },
    {
      "class": "smoke",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002434"
    },
    {
      "class": "fire",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_test_fire000837"
    },
    {
      "class": "fire",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002294"
    },
    {
      "class": "fire",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002301"
    },
    {
      "class": "fire",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002309"
    },
    {
      "class": "fire",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002325"
    },
    {
      "class": "fire",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002342"
    },
    {
      "class": "fire",
      "reason": "pathological_tiny_miss",
      "sample_id": "yingjie_train_fire002374"
    }
  ],
  "questionable_positives_excluded": 50,
  "status": "PASS",
  "thresholds": {
    "final_pred_conf": 0.25,
    "near_boundary_floor": 0.1,
    "source": "v2_2_workflow.mine_error_rows conf=0.25; mine_error_rows_real conf=0.10"
  },
  "v2_1_train_negatives_mined": 800,
  "v2_1_train_negatives_scanned": 800,
  "yingjie_eligible_negatives": 22
}
```
