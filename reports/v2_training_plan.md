# Dataset V2 Training Plan

This plan details the experimental sequence for training lightweight models on the newly constructed Dataset V2 for CPU inference.

## Objective
The primary goal is to maximize mAP while ensuring average CPU inference latency stays below 30ms on the target SecureVU architecture, drastically reducing false positives (by utilizing the V2 hard negatives), and improving aggregate recall for both fire and smoke.

## Data Baseline
- **Dataset**: `data/processed/fire_smoke_v2`
- **Classes**: 0=fire, 1=smoke
- **Test Set**: The V2 test split is explicitly untouched and must NOT be used for threshold tuning or architecture selection. It is reserved for the final production gate.

## Planned Experiments

### Experiment A: YOLO11n Baseline Continuity
- **Model**: YOLO11n
- **Image Size**: 512
- **Epochs**: 80 maximum (with early stopping patience=15)
- **Purpose**: Establishes a direct comparison against the V0 baseline to quantify the performance uplift strictly from the V2 dataset (particularly the inclusion of hard negatives and smoke boosters).

### Experiment B: YOLOX-Nano (416)
- **Model**: YOLOX-Nano
- **Image Size**: 416
- **Epochs**: 100
- **Purpose**: Target ultra-fast CPU inference latency. YOLOX's anchor-free design may offer better generalization on small smoke puffs at low resolutions.

### Experiment C: YOLOX-Nano (512)
- **Model**: YOLOX-Nano
- **Image Size**: 512
- **Epochs**: 100
- **Purpose**: Compare size/speed trade-offs against Experiment B to determine if the resolution bump justifies the latency penalty for small fire detection.

### Experiment D (Optional): RTMDet-tiny
- **Model**: RTMDet-tiny
- **Image Size**: 512
- **Purpose**: Alternative architecture known for excellent CPU throughput and robust bounding boxes.

## Selection Metrics
Do not optimize solely for mAP. The winning model will be selected based on a balanced scorecard of:
- **Accuracy**: mAP50, mAP50-95, Fire Precision/Recall, Smoke Precision/Recall.
- **Reliability**: False detections on negative images, negative-image false-positive rate.
- **Practical Application**: Event recall (video level), false alarms per video-hour, time to first detection.
- **Hardware constraints**: Average CPU latency, p95 CPU latency, Peak RAM usage, Model size in MB.
