#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_smoke_cpu.annotations import parse_voc_xml, write_yolo_label
from fire_smoke_cpu.constants import RAW_DIR, REPORT_DIR
from fire_smoke_cpu.prep import build_manifest_row, canonical_paths, materialize_canonical_image, merge_all_manifests, write_source_manifest
from fire_smoke_cpu.utils import ensure_dirs, image_size, list_images, utc_now_iso, write_json


def find_xml(image: Path, root: Path) -> Path | None:
    candidates = [
        image.with_suffix(".xml"),
        Path(str(image).replace("/JPEGImages/", "/Annotations/")).with_suffix(".xml"),
        Path(str(image).replace("/images/", "/annotations/")).with_suffix(".xml"),
        Path(str(image).replace("/Images/", "/Annotations/")).with_suffix(".xml"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(root.rglob(f"{image.stem}.xml"))
    return matches[0] if matches else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR / "ms_fsdb")
    args = parser.parse_args()
    ensure_dirs(REPORT_DIR)
    images = list_images(args.raw_dir)
    rows = []
    exclusions = []
    unknown_classes = {}
    for image in images:
        xml = find_xml(image, args.raw_dir)
        boxes = []
        errors = []
        metadata = {"unknown_classes": []}
        if xml:
            try:
                size = image_size(image)
            except Exception:
                size = None
            boxes, errors, metadata = parse_voc_xml(xml, fallback_size=size)
        canonical_image, canonical_label, _sample_id = canonical_paths("ms_fsdb", image)
        try:
            materialize_canonical_image(image, canonical_image)
            for name in metadata.get("unknown_classes", []):
                unknown_classes[name] = unknown_classes.get(name, 0) + 1
            severe = [e for e in errors if "unknown_class" in e or "invalid_bbox" in e or "parse" in e or "size" in e]
            if severe:
                exclusions.append({"image": str(image), "label": str(xml or ""), "errors": errors})
                continue
            write_yolo_label(canonical_label, boxes)
            rows.append(
                build_manifest_row(
                    source_dataset="ms_fsdb",
                    image_path=canonical_image,
                    label_path=canonical_label,
                    original_image_path=image,
                    original_label_path=xml,
                    boxes=boxes,
                    license_status="UNKNOWN",
                )
            )
        except Exception as exc:
            exclusions.append({"image": str(image), "label": str(xml or ""), "errors": [repr(exc)]})
    write_source_manifest("ms_fsdb", rows)
    merge_all_manifests()
    report = {
        "timestamp_utc": utc_now_iso(),
        "raw_dir": str(args.raw_dir),
        "images_discovered": len(images),
        "valid_samples": len(rows),
        "excluded": len(exclusions),
        "unknown_classes": unknown_classes,
        "exclusions": exclusions[:1000],
    }
    write_json(REPORT_DIR / "ms_fsdb_quality_report.json", report)
    md = ["# MS-FSDB Quality Report", "", f"- Images discovered: `{len(images)}`", f"- Valid samples: `{len(rows)}`", f"- Excluded: `{len(exclusions)}`", f"- Unknown classes: `{unknown_classes}`", ""]
    if exclusions:
        md.append("See JSON report for exclusion details.")
    (REPORT_DIR / "ms_fsdb_quality_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if images else 2


if __name__ == "__main__":
    raise SystemExit(main())
