# V0 vs V2 Model Comparison

## Overall Metrics (Validation Set)

| Metric | V0 Baseline | V2 Fine-Tuned |
|---|---|---|
| metrics/mAP50(B) | 0.5557 | 0.4685 |
| metrics/mAP50-95(B) | 0.2909 | 0.2516 |
| metrics/precision(B) | 0.7280 | 0.7947 |
| metrics/recall(B) | 0.4920 | 0.3984 |

## File Size & Hardware

- **V0 Model Size:** ~5.2 MB
- **V2 Model Size:** ~5.2 MB (Same architecture)

*Note: Class-specific metrics and negative false-positive rates require running the full evaluate_video.py script on test videos.*