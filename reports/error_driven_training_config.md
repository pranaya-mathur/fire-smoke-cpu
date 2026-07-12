# Error-Driven Training Config

```json
{
  "batch": 4,
  "cache": false,
  "device": "cpu",
  "epochs": 1,
  "freeze_modules": 21,
  "freeze_strategy": "Ultralytics freeze=21: train Detect head + last 2-3 neck modules only",
  "frozen_parameter_count": 1780288,
  "imgsz": 512,
  "lr0": 5e-05,
  "lrf": 0.01,
  "model": "yolo11n",
  "module_count": 24,
  "module_info": [
    {
      "index": 0,
      "type": "Conv"
    },
    {
      "index": 1,
      "type": "Conv"
    },
    {
      "index": 2,
      "type": "C3k2"
    },
    {
      "index": 3,
      "type": "Conv"
    },
    {
      "index": 4,
      "type": "C3k2"
    },
    {
      "index": 5,
      "type": "Conv"
    },
    {
      "index": 6,
      "type": "C3k2"
    },
    {
      "index": 7,
      "type": "Conv"
    },
    {
      "index": 8,
      "type": "C3k2"
    },
    {
      "index": 9,
      "type": "SPPF"
    },
    {
      "index": 10,
      "type": "C2PSA"
    },
    {
      "index": 11,
      "type": "Upsample"
    },
    {
      "index": 12,
      "type": "Concat"
    },
    {
      "index": 13,
      "type": "C3k2"
    },
    {
      "index": 14,
      "type": "Upsample"
    },
    {
      "index": 15,
      "type": "Concat"
    },
    {
      "index": 16,
      "type": "C3k2"
    },
    {
      "index": 17,
      "type": "Conv"
    },
    {
      "index": 18,
      "type": "Concat"
    },
    {
      "index": 19,
      "type": "C3k2"
    },
    {
      "index": 20,
      "type": "Conv"
    },
    {
      "index": 21,
      "type": "Concat"
    },
    {
      "index": 22,
      "type": "C3k2"
    },
    {
      "index": 23,
      "type": "Detect"
    }
  ],
  "optimizer": "AdamW",
  "output_project": "runs/detect",
  "parameter_name_sample": [
    {
      "name": "model.0.conv.weight",
      "numel": 432,
      "requires_grad": false
    },
    {
      "name": "model.0.bn.weight",
      "numel": 16,
      "requires_grad": false
    },
    {
      "name": "model.0.bn.bias",
      "numel": 16,
      "requires_grad": false
    },
    {
      "name": "model.1.conv.weight",
      "numel": 4608,
      "requires_grad": false
    },
    {
      "name": "model.1.bn.weight",
      "numel": 32,
      "requires_grad": false
    },
    {
      "name": "model.1.bn.bias",
      "numel": 32,
      "requires_grad": false
    },
    {
      "name": "model.2.cv1.conv.weight",
      "numel": 1024,
      "requires_grad": false
    },
    {
      "name": "model.2.cv1.bn.weight",
      "numel": 32,
      "requires_grad": false
    },
    {
      "name": "model.2.cv1.bn.bias",
      "numel": 32,
      "requires_grad": false
    },
    {
      "name": "model.2.cv2.conv.weight",
      "numel": 3072,
      "requires_grad": false
    },
    {
      "name": "model.2.cv2.bn.weight",
      "numel": 64,
      "requires_grad": false
    },
    {
      "name": "model.2.cv2.bn.bias",
      "numel": 64,
      "requires_grad": false
    },
    {
      "name": "model.2.m.0.cv1.conv.weight",
      "numel": 1152,
      "requires_grad": false
    },
    {
      "name": "model.2.m.0.cv1.bn.weight",
      "numel": 8,
      "requires_grad": false
    },
    {
      "name": "model.2.m.0.cv1.bn.bias",
      "numel": 8,
      "requires_grad": false
    }
  ],
  "run_name": "v2_1_error_driven_replay_challenger_1e",
  "seed": 42,
  "starting_checkpoint": "runs/detect/runs/detect/yolo11n_v2_1_real_512_12e/weights/best.pt",
  "trainable_parameter_count": 809942,
  "workers": 2
}
```
