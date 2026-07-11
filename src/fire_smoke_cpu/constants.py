from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_DIR = DATA_DIR / "manifests"
REPORT_DIR = ROOT / "reports"

CLASS_NAME_TO_ID = {
    "fire": 0,
    "smoke": 1,
}
CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}

MANIFEST_COLUMNS = [
    "sample_id",
    "canonical_image_path",
    "canonical_label_path",
    "source_dataset",
    "original_image_path",
    "original_label_path",
    "original_filename",
    "sha256",
    "perceptual_hash",
    "width",
    "height",
    "has_fire",
    "has_smoke",
    "fire_box_count",
    "smoke_box_count",
    "is_negative",
    "group_id",
    "scene_id",
    "video_id",
    "license_status",
    "split",
    "excluded",
    "exclusion_reason",
]
