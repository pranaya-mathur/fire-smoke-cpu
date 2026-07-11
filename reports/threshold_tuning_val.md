# Threshold Tuning (val)

- Model: `runs/detect/runs/yolo11n_hf_indoor_512/cpu_20e/weights/best.pt`
- Images: `772`
- IoU match threshold: `0.5`
- Recommended fire threshold: `0.5`
- Recommended smoke threshold: `0.3`

| Threshold | Fire precision | Fire recall | Fire FP/img | Smoke precision | Smoke recall | Smoke FP/img |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.350 | 0.780 | 1.269 | 0.259 | 0.566 | 0.557 |
| 0.10 | 0.438 | 0.742 | 0.834 | 0.322 | 0.513 | 0.370 |
| 0.15 | 0.510 | 0.719 | 0.605 | 0.368 | 0.464 | 0.273 |
| 0.20 | 0.570 | 0.699 | 0.462 | 0.409 | 0.430 | 0.214 |
| 0.25 | 0.623 | 0.685 | 0.364 | 0.456 | 0.408 | 0.167 |
| 0.30 | 0.682 | 0.672 | 0.275 | 0.505 | 0.389 | 0.131 |
| 0.35 | 0.745 | 0.656 | 0.197 | 0.530 | 0.366 | 0.111 |
| 0.40 | 0.806 | 0.648 | 0.137 | 0.538 | 0.347 | 0.102 |
| 0.50 | 0.888 | 0.620 | 0.069 | 0.631 | 0.309 | 0.062 |
| 0.60 | 0.936 | 0.539 | 0.032 | 0.699 | 0.245 | 0.036 |
| 0.70 | 0.973 | 0.427 | 0.010 | 0.754 | 0.196 | 0.022 |

False-positive rates here are unmatched predictions on labeled validation/test images. A real false-alarm gate still needs negative CCTV clips/images.
