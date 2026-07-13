# License Manifest

This file tracks license-discovery status. It is not legal advice.

| Name | Source URL | License discovered | License file URL | Commercial use status | Notes |
|---|---|---|---|---|---|
| D-Fire dataset | https://github.com/gaia-solutions-on-demand/DFireDataset | CC0-1.0 in official LICENSE | https://github.com/gaia-solutions-on-demand/DFireDataset/blob/main/LICENSE | confirmed by license text; still review in commercial release process | Images/labels via official README Kaggle mirror (`sayedgamal99/smoke-fire-detection-yolo`). |
| FIRESENSE | https://zenodo.org/records/836749 | UNKNOWN |  | requires review | Temporal / video validation source. |
| HF Kien CCTV archive | https://huggingface.co/datasets/KienNgyuen/Fire-Smoke-Detection | CONFLICT: HF repo Apache-2.0 metadata; matching Simuletic dataset page shows cc-by-nc-4.0 tag and CC BY 4.0 card text |  | requires review | Downloaded only for inspection; do not use for commercial training until conflict is resolved. |
| HF Kien Indoor Fire Smoke archive | https://huggingface.co/datasets/KienNgyuen/Fire-Smoke-Detection | Roboflow YAML says CC BY 4.0 |  | requires review | Class names are numeric; visual mapping is 0 fire / 1 smoke but source confirmation is needed. |
| Ultralytics code/model | https://github.com/ultralytics/ultralytics | UNKNOWN |  | requires review | Review installed package and model weight terms before commercial use. |
| YOLOX code | https://github.com/Megvii-BaseDetection/YOLOX | UNKNOWN |  | requires review | Challenger plan only until baseline completes. |
| YOLOX pretrained weights | https://github.com/Megvii-BaseDetection/YOLOX | UNKNOWN |  | requires review | Track separately from code license. |
