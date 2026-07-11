#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_smoke_cpu.annotations import parse_yolo_label, write_yolo_label
from fire_smoke_cpu.constants import RAW_DIR, REPORT_DIR
from fire_smoke_cpu.prep import build_manifest_row, canonical_paths, materialize_canonical_image, merge_all_manifests, write_source_manifest
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR / "dfire")
    args = parser.parse_args()
    ensure_dirs(REPORT_DIR)
    images = list_images(args.raw_dir)
    rows = []
    exclusions = []
    for image in images:
        label = find_label(image, args.raw_dir)
        boxes = []
        errors = []
        if label:
            boxes, errors = parse_yolo_label(label)
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
                    license_status="UNKNOWN",
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
        "exclusions": exclusions[:1000],
    }
    write_json(REPORT_DIR / "dfire_quality_report.json", report)
    md = ["# D-Fire Quality Report", "", f"- Images discovered: `{len(images)}`", f"- Valid samples: `{len(rows)}`", f"- Excluded: `{len(exclusions)}`", ""]
    if exclusions:
        md.append("See JSON report for exclusion details.")
    (REPORT_DIR / "dfire_quality_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if images else 2


if __name__ == "__main__":
    raise SystemExit(main())
