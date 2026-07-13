#!/usr/bin/env python3
"""Prepare official D-Fire YOLO images/labels into canonical manifests.

SecureVU canonical classes are always 0=fire, 1=smoke.

The official D-Fire README Kaggle mirror
(`sayedgamal99/smoke-fire-detection-yolo`) ships data.yaml with
names: ['smoke', 'fire'] (0=smoke, 1=fire). That ordering is swapped
relative to SecureVU and is remapped automatically when detected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_smoke_cpu.annotations import YoloBox, parse_yolo_label, write_yolo_label
from fire_smoke_cpu.constants import RAW_DIR, REPORT_DIR
from fire_smoke_cpu.prep import (
    build_manifest_row,
    canonical_paths,
    materialize_canonical_image,
    merge_all_manifests,
    write_source_manifest,
)
from fire_smoke_cpu.utils import ensure_dirs, list_images, utc_now_iso, write_json


def find_label(image: Path, root: Path) -> Path | None:
    candidates = [
        image.with_suffix(".txt"),
        Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt"),
        root / "labels" / image.relative_to(root).with_suffix(".txt").name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(root.rglob(f"{image.stem}.txt"))
    return matches[0] if matches else None


def detect_source_class_order(raw_dir: Path) -> list[str] | None:
    """Return YOLO names list from nearby data.yaml if present."""
    candidates = [
        raw_dir / "data.yaml",
        raw_dir / "kaggle_smoke_fire_detection_yolo" / "data.yaml",
    ]
    for yaml_path in candidates:
        if not yaml_path.exists():
            continue
        text = yaml_path.read_text(encoding="utf-8", errors="replace")
        # Minimal parse: names: ['smoke', 'fire']
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("names:"):
                lower = stripped.lower()
                if "smoke" in lower and "fire" in lower:
                    # order by first occurrence in the list literal
                    smoke_i = lower.find("smoke")
                    fire_i = lower.find("fire")
                    if smoke_i < fire_i:
                        return ["smoke", "fire"]
                    return ["fire", "smoke"]
    return None


def remap_boxes(boxes: list[YoloBox], source_names: list[str] | None) -> list[YoloBox]:
    """Map source class IDs onto SecureVU 0=fire, 1=smoke."""
    if not source_names:
        return boxes
    name_to_src = {name: idx for idx, name in enumerate(source_names)}
    src_to_canonical = {}
    if "fire" in name_to_src:
        src_to_canonical[name_to_src["fire"]] = 0
    if "smoke" in name_to_src:
        src_to_canonical[name_to_src["smoke"]] = 1
    if src_to_canonical == {0: 0, 1: 1}:
        return boxes
    remapped = []
    for box in boxes:
        new_id = src_to_canonical.get(box.class_id)
        if new_id is None:
            remapped.append(box)
            continue
        remapped.append(
            YoloBox(
                class_id=new_id,
                x_center=box.x_center,
                y_center=box.y_center,
                width=box.width,
                height=box.height,
            )
        )
    return remapped


def is_meaningful_image(image: Path) -> bool:
    parts = {part.lower() for part in image.parts}
    if "figures" in parts:
        return False
    if image.name.lower() == "dfire_examples.png":
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR / "dfire")
    parser.add_argument(
        "--swap-classes",
        action="store_true",
        help="Force remap assuming source names are [smoke, fire] (0=smoke, 1=fire).",
    )
    args = parser.parse_args()
    ensure_dirs(REPORT_DIR)

    source_names = ["smoke", "fire"] if args.swap_classes else detect_source_class_order(args.raw_dir)
    images = [p for p in list_images(args.raw_dir) if is_meaningful_image(p)]
    rows = []
    exclusions = []
    remapped_count = 0
    for image in images:
        label = find_label(image, args.raw_dir)
        boxes = []
        errors = []
        if label:
            boxes, errors = parse_yolo_label(label)
            if boxes and source_names and source_names != ["fire", "smoke"]:
                boxes = remap_boxes(boxes, source_names)
                remapped_count += 1
        else:
            errors = ["missing_label"]
        canonical_image, canonical_label, _sample_id = canonical_paths("dfire", image)
        try:
            materialize_canonical_image(image, canonical_image)
            if errors:
                exclusions.append({"image": str(image), "label": str(label or ""), "errors": errors})
                continue
            write_yolo_label(canonical_label, boxes)
            rows.append(
                build_manifest_row(
                    source_dataset="dfire",
                    image_path=canonical_image,
                    label_path=canonical_label,
                    original_image_path=image,
                    original_label_path=label,
                    boxes=boxes,
                    license_status="CC0-1.0",
                )
            )
        except Exception as exc:
            exclusions.append({"image": str(image), "label": str(label or ""), "errors": [repr(exc)]})
    write_source_manifest("dfire", rows)
    merge_all_manifests()
    report = {
        "timestamp_utc": utc_now_iso(),
        "raw_dir": str(args.raw_dir),
        "images_discovered": len(images),
        "valid_samples": len(rows),
        "excluded": len(exclusions),
        "source_class_names": source_names,
        "canonical_class_names": ["fire", "smoke"],
        "labels_remapped_to_canonical": remapped_count,
        "class_remap_applied": bool(source_names and source_names != ["fire", "smoke"]),
        "exclusions": exclusions[:1000],
        "notes": (
            "Official D-Fire README Kaggle mirror. "
            "Raw archives untouched; remapping applied only to canonical labels."
        ),
    }
    write_json(REPORT_DIR / "dfire_quality_report.json", report)
    md = [
        "# D-Fire Quality Report",
        "",
        f"- Images discovered: `{len(images)}`",
        f"- Valid samples: `{len(rows)}`",
        f"- Excluded: `{len(exclusions)}`",
        f"- Source class names: `{source_names}`",
        f"- Class remap applied: `{report['class_remap_applied']}`",
        f"- Labels remapped: `{remapped_count}`",
        "",
    ]
    if exclusions:
        md.append("See JSON report for exclusion details.")
    (REPORT_DIR / "dfire_quality_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "exclusions"}, indent=2))
    return 0 if images else 2


if __name__ == "__main__":
    raise SystemExit(main())
