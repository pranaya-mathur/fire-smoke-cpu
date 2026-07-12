# SecureVU Real Dataset V2.1 Completion Report

- Commit: b5893d7d899ef62ac8d4f6229bc9f746f0eb4371
- Branch: main
- Git status entries: 111
- HF auth: PASS as chinki-m
- Quality gate: PASS
- NO_MOCK_DATA_FOR_REAL_TRAINING: True
- Best model: runs/detect/runs/detect/yolo11n_v2_1_real_512_12e/weights/best.pt
- Epochs completed: 6 of max 12, patience 3
- Best epoch: 3, mAP50 0.5834, mAP50-95 0.3272

## Downloads
- dfire: NOT_DOWNLOADED, images 0, labels 0, size 0, rev None
- libreyolo: DOWNLOAD_COMPLETE, images 746, labels 748, size 43726053, rev fcc82783b1a61a87ab0bfed5b3727b344f2236a5
- medyoussef: DOWNLOAD_COMPLETE, images 22327, labels 22327, size 3242604105, rev 5db7b603eebc0229ac9fbff1bce760ea97bd2f5d

## Dataset
- Samples: 8656
- Splits: {'test': 1172, 'train': 6370, 'val': 1114}
- Sources: {'LibreYOLO/smoke-uvylj': 741, 'hf_kien_indoor_existing': 4915, 'medyoussef/fire-smoke-hardnegatives-int8': 3000}
- Medyoussef status counts: {'ANNOTATED_POSITIVE': 11349, 'CONFIRMED_NEGATIVE': 10638, 'UNKNOWN': 340}
- LibreYOLO status counts: {'UNKNOWN_CLASS_OR_INVALID_ANNOTATION': 5, 'VALID_SMOKE': 741}
- Exact duplicate groups: 0
- Near duplicate pairs: 3579703
- Label conflicts: 0

## Test Metrics
| Model | mAP50 | mAP50-95 | Fire mAP50 | Smoke mAP50 |
|---|---:|---:|---:|---:|
| v0 | 0.5547 | 0.2717 | 0.6592 | 0.4502 |
| v2 | 0.4387 | 0.2052 | 0.5746 | 0.3027 |
| real_v2_1 | 0.5751 | 0.2750 | 0.6346 | 0.5156 |

## Negative Evaluation
| Model | FP image rate | False fire imgs | False smoke imgs | FP/image |
|---|---:|---:|---:|---:|
| v0 | 0.3489 | 58 | 42 | 0.6043 |
| v2 | 0.2590 | 45 | 29 | 0.4388 |
| real_v2_1 | 0.0468 | 12 | 1 | 0.0647 |

## CPU Benchmark
| Model | p95 ms | Avg RSS MB | Peak RSS MB | Size MB |
|---|---:|---:|---:|---:|
| v0 | 51.49 | 415.2 | 416.0 | 5.20 |
| v2 | 43.69 | 426.6 | 427.0 | 5.20 |
| real_v2_1 | 44.99 | 437.5 | 437.6 | 5.20 |

## Threshold Tuning
- Negative-only best validation pair: fire 0.7, smoke 0.7, FP image rate 0.0000
- Note: this threshold sweep measured validation negatives only; choose production thresholds with positive recall constraints before deployment.

## Reproduction Commands
```bash
./.venv/bin/python scripts/real_v2_1_workflow.py --phase all
pytest -q
./.venv/bin/python scripts/smoke_test_v2_1.py
./.venv/bin/python scripts/train_v2_1.py
```
