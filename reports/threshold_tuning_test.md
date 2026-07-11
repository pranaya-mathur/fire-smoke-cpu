# Threshold Tuning (test)

- Model: `runs/detect/runs/yolo11n_hf_indoor_512/cpu_20e/weights/best.pt`
- Images: `824`
- IoU match threshold: `0.5`
- Recommended fire threshold: `0.35`
- Recommended smoke threshold: `0.4`

| Threshold | Fire precision | Fire recall | Fire FP/img | Smoke precision | Smoke recall | Smoke FP/img |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.322 | 0.836 | 0.933 | 0.282 | 0.603 | 1.380 |
| 0.10 | 0.426 | 0.799 | 0.572 | 0.356 | 0.552 | 0.899 |
| 0.15 | 0.502 | 0.774 | 0.408 | 0.411 | 0.525 | 0.677 |
| 0.20 | 0.561 | 0.747 | 0.311 | 0.454 | 0.501 | 0.542 |
| 0.25 | 0.618 | 0.724 | 0.238 | 0.486 | 0.484 | 0.460 |
| 0.30 | 0.656 | 0.696 | 0.194 | 0.525 | 0.468 | 0.381 |
| 0.35 | 0.701 | 0.674 | 0.153 | 0.572 | 0.457 | 0.308 |
| 0.40 | 0.735 | 0.639 | 0.123 | 0.605 | 0.440 | 0.258 |
| 0.50 | 0.801 | 0.580 | 0.076 | 0.669 | 0.401 | 0.178 |
| 0.60 | 0.843 | 0.489 | 0.049 | 0.726 | 0.347 | 0.118 |
| 0.70 | 0.890 | 0.352 | 0.023 | 0.831 | 0.285 | 0.052 |

False-positive rates here are unmatched predictions on labeled validation/test images. A real false-alarm gate still needs negative CCTV clips/images.
