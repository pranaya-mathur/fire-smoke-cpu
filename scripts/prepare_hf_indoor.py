#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_smoke_cpu.annotations import parse_yolo_label, write_yolo_label
from fire_smoke_cpu.constants import RAW_DIR, REPORT_DIR
from fire_smoke_cpu.prep import build_manifest_row, canonical_paths, materialize_canonical_image, merge_all_manifests, write_source_manifest
from fire_smoke_cpu.utils import ensure_dirs, list_images, utc_now_iso, write_json


SOURCE_DATASET = "hf_kien_indoor"
LICENSE_STATUS = "requires review; Roboflow YAML says CC BY 4.0; class mapping visually verified"


def find_dataset_root(raw_dir: Path) -> Path | None:
    candidates = [
        raw_dir,
        raw_dir / "Indoor Fire Smoke",
        raw_dir / "extracted_indoor" / "Indoor Fire Smoke",
    ]
    for candidate in candidates:
        if (candidate / "data.yaml").exists() and (candidate / "train" / "images").exists():
            return candidate
    for candidate in raw_dir.rglob("data.yaml"):
        parent = candidate.parent
        if "__MACOSX" not in parent.parts and (parent / "train" / "images").exists():
            return parent
    return None


def raw_split_for(image: Path, dataset_root: Path) -> str:
    try:
        relative = image.relative_to(dataset_root)
    except ValueError:
        return ""
    return relative.parts[0] if relative.parts else ""


def label_for(image: Path, dataset_root: Path) -> Path:
    split = raw_split_for(image, dataset_root)
    return dataset_root / split / "labels" / f"{image.stem}.txt"


def roboflow_base_stem(stem: str) -> str:
    base = stem.split(".rf.", 1)[0]
    for suffix in ("_jpg", "_jpeg", "_png", "_bmp", "_webp"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def group_id_for(image: Path, dataset_root: Path) -> tuple[str, str, str]:
    base = roboflow_base_stem(image.stem)
    lowered = base.lower()
    video_id = ""

    video_match = re.match(r"(.+?_mp4)-\d+$", lowered)
    if video_match:
        video_id = video_match.group(1)
        return f"{SOURCE_DATASET}:video:{video_id}", base, video_id

    numbered_prefix = re.match(r"([a-z]+)[_-]\d{1,7}$", lowered)
    if numbered_prefix:
        scene_id = numbered_prefix.group(1)
        return f"{SOURCE_DATASET}:sequence:{scene_id}", scene_id, video_id

    fire_tlm_match = re.match(r"(fire_tlm_[a-z0-9]+?)\d+$", lowered)
    if fire_tlm_match:
        scene_id = fire_tlm_match.group(1)
        return f"{SOURCE_DATASET}:sequence:{scene_id}", scene_id, video_id

    return f"{SOURCE_DATASET}:image:{lowered}", base, video_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR / "hf_kien_fire_smoke" / "extracted_indoor")
    args = parser.parse_args()
    ensure_dirs(REPORT_DIR)

    dataset_root = find_dataset_root(args.raw_dir)
    if dataset_root is None:
        print(f"Could not find HF Indoor dataset root under {args.raw_dir}")
        return 2

    images = [image for image in list_images(dataset_root) if "__MACOSX" not in image.parts and "/images/" in image.as_posix()]
    rows = []
    exclusions = []
    raw_split_counts = Counter()
    class_counts = Counter()
    group_counts = Counter()

    for image in images:
        raw_split = raw_split_for(image, dataset_root)
        raw_split_counts[raw_split] += 1
        label = label_for(image, dataset_root)
        boxes = []
        errors = []
        if label.exists():
            boxes, errors = parse_yolo_label(label)
        else:
            errors = ["missing_label"]

        for box in boxes:
            class_counts[str(box.class_id)] += 1

        canonical_image, canonical_label, _sample_id = canonical_paths(SOURCE_DATASET, image)
        group_id, scene_id, video_id = group_id_for(image, dataset_root)
        group_counts[group_id] += 1
        try:
            materialize_canonical_image(image, canonical_image)
            if errors:
                exclusions.append({"image": str(image), "label": str(label), "errors": errors})
                continue
            write_yolo_label(canonical_label, boxes)
            rows.append(
                build_manifest_row(
                    source_dataset=SOURCE_DATASET,
                    image_path=canonical_image,
                    label_path=canonical_label,
                    original_image_path=image,
                    original_label_path=label,
                    boxes=boxes,
                    license_status=LICENSE_STATUS,
                    group_id=group_id,
                    scene_id=scene_id,
                    video_id=video_id,
                )
            )
        except Exception as exc:
            exclusions.append({"image": str(image), "label": str(label), "errors": [repr(exc)]})

    write_source_manifest(SOURCE_DATASET, rows)
    merge_all_manifests()

    large_groups = {group: count for group, count in group_counts.items() if count >= 25}
    report = {
        "timestamp_utc": utc_now_iso(),
        "raw_dir": str(args.raw_dir),
        "dataset_root": str(dataset_root),
        "images_discovered": len(images),
        "valid_samples": len(rows),
        "excluded": len(exclusions),
        "raw_split_counts": dict(raw_split_counts),
        "box_counts_by_raw_class_id": dict(class_counts),
        "group_count": len(group_counts),
        "large_groups": large_groups,
        "class_mapping": {"0": "fire", "1": "smoke"},
        "license_status": LICENSE_STATUS,
        "exclusions": exclusions[:1000],
    }
    write_json(REPORT_DIR / "hf_indoor_quality_report.json", report)
    md = [
        "# HF Indoor Quality Report",
        "",
        "HF Indoor is the active fallback V1 source after official D-Fire/MS-FSDB downloads were blocked.",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- Images discovered: `{len(images)}`",
        f"- Valid samples: `{len(rows)}`",
        f"- Excluded: `{len(exclusions)}`",
        f"- Raw split counts: `{dict(raw_split_counts)}`",
        f"- Box counts by raw class id: `{dict(class_counts)}`",
        f"- Canonical class mapping: `0=fire`, `1=smoke`",
        f"- Inferred group count: `{len(group_counts)}`",
        f"- Groups with >=25 images: `{large_groups}`",
        "",
        "The provider's original train/valid/test split is recorded but not trusted for final training; the project split step rebuilds leakage-aware splits from canonical manifests.",
        "",
        "License note: Roboflow YAML says CC BY 4.0, but commercial approval still requires review. This report is not legal advice.",
    ]
    if exclusions:
        md.extend(["", "See JSON report for exclusion details."])
    (REPORT_DIR / "hf_indoor_quality_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
