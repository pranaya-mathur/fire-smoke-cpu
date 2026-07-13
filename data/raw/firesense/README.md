# FIRESENSE (Zenodo record 836749)

Official direct ZIP downloads. Raw archives are immutable once downloaded.

| Archive | URL | Contents |
|---|---|---|
| `fire_videos.1406.zip` | https://zenodo.org/records/836749/files/fire_videos.1406.zip?download=1 | 11 positive + 16 negative flame videos (~623 MB) |
| `smoke_videos.1407.zip` | https://zenodo.org/records/836749/files/smoke_videos.1407.zip?download=1 | 13 positive + 9 negative smoke videos (~198 MB) |

```bash
python scripts/download_priority_sources.py download --firesense
```

Extracted trees land under `extracted/fire_videos/` and `extracted/smoke_videos/`.

**Annotation note:** FIRESENSE is video-only here. It does **not** provide YOLO boxes out of the box.
Qualification marks `object_detection_annotations=False` until a frame+label pipeline exists.
Use for temporal/eval clips; do not blind-merge into V2.1 training.
