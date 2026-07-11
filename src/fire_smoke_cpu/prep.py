from __future__ import annotations

import re
from pathlib import Path

from .annotations import YoloBox
from .constants import MANIFEST_COLUMNS, MANIFEST_DIR, PROCESSED_DIR
from .utils import ensure_dirs, open_image, perceptual_hash, safe_link_or_copy, sha256_file, write_csv_dicts


def infer_group_id(path: Path) -> str:
    stem = path.stem.lower()
    parent = path.parent.name.lower()
    tokens = re.split(r"[_\-\s.]+", stem)
    if len(tokens) > 1:
        return f"{parent}:{tokens[0]}"
    return f"{parent}:{stem}"


def sample_flags(boxes: list[YoloBox]) -> dict:
    fire_count = sum(1 for box in boxes if box.class_id == 0)
    smoke_count = sum(1 for box in boxes if box.class_id == 1)
    return {
        "has_fire": fire_count > 0,
        "has_smoke": smoke_count > 0,
        "fire_box_count": fire_count,
        "smoke_box_count": smoke_count,
        "is_negative": fire_count == 0 and smoke_count == 0,
    }


def canonical_paths(source_dataset: str, image: Path) -> tuple[Path, Path, str]:
    sample_id = f"{source_dataset}_{image.stem}_{sha256_file(image)[:12]}"
    suffix = image.suffix.lower() if image.suffix else ".jpg"
    image_path = PROCESSED_DIR / "canonical" / source_dataset / "images" / f"{sample_id}{suffix}"
    label_path = PROCESSED_DIR / "canonical" / source_dataset / "labels" / f"{sample_id}.txt"
    return image_path, label_path, sample_id


def build_manifest_row(
    *,
    source_dataset: str,
    image_path: Path,
    label_path: Path,
    original_image_path: Path,
    original_label_path: Path | None,
    boxes: list[YoloBox],
    license_status: str,
    excluded: bool = False,
    exclusion_reason: str = "",
    group_id: str | None = None,
    scene_id: str = "",
    video_id: str = "",
) -> dict:
    with open_image(image_path) as image:
        width, height = image.size
    flags = sample_flags(boxes)
    return {
        "sample_id": image_path.stem,
        "canonical_image_path": str(image_path),
        "canonical_label_path": str(label_path),
        "source_dataset": source_dataset,
        "original_image_path": str(original_image_path),
        "original_label_path": str(original_label_path or ""),
        "original_filename": original_image_path.name,
        "sha256": sha256_file(image_path),
        "perceptual_hash": perceptual_hash(image_path),
        "width": width,
        "height": height,
        **flags,
        "group_id": group_id or infer_group_id(original_image_path),
        "scene_id": scene_id,
        "video_id": video_id,
        "license_status": license_status,
        "split": "",
        "excluded": excluded,
        "exclusion_reason": exclusion_reason,
    }


def write_source_manifest(source_dataset: str, rows: list[dict]) -> Path:
    ensure_dirs(MANIFEST_DIR)
    path = MANIFEST_DIR / f"{source_dataset}_samples.csv"
    write_csv_dicts(path, rows, MANIFEST_COLUMNS)
    return path


def merge_all_manifests() -> Path:
    rows: list[dict] = []
    for path in sorted(MANIFEST_DIR.glob("*_samples.csv")):
        if path.name == "all_samples.csv":
            continue
        import csv

        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    out = MANIFEST_DIR / "all_samples.csv"
    write_csv_dicts(out, rows, MANIFEST_COLUMNS)
    return out


def materialize_canonical_image(src: Path, dst: Path) -> str:
    return safe_link_or_copy(src, dst)
