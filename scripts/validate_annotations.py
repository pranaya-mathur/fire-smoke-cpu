#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_smoke_cpu.annotations import parse_yolo_label
from fire_smoke_cpu.constants import MANIFEST_DIR, REPORT_DIR
from fire_smoke_cpu.utils import open_image, write_csv_dicts


FIELDS = ["sample_id", "image_path", "label_path", "severity", "reason"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "all_samples.csv")
    args = parser.parse_args()
    invalid = []
    if not args.manifest.exists():
        print(f"Missing manifest: {args.manifest}")
        return 2
    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8")))
    known_labels = set()
    for row in rows:
        image_path = Path(row["canonical_image_path"])
        label_path = Path(row["canonical_label_path"])
        known_labels.add(label_path)
        try:
            with open_image(image_path) as image:
                if image.width <= 0 or image.height <= 0:
                    invalid.append({**row, "image_path": str(image_path), "label_path": str(label_path), "severity": "severe", "reason": "invalid_image_dimensions"})
        except Exception as exc:
            invalid.append({**row, "image_path": str(image_path), "label_path": str(label_path), "severity": "severe", "reason": f"image_decode_error:{exc}"})
            continue
        boxes, errors = parse_yolo_label(label_path)
        for error in errors:
            invalid.append({**row, "image_path": str(image_path), "label_path": str(label_path), "severity": "severe", "reason": error})
        if row.get("is_negative") in {"True", "true", "1"} and boxes:
            invalid.append({**row, "image_path": str(image_path), "label_path": str(label_path), "severity": "severe", "reason": "negative_has_boxes"})

    for label in (ROOT / "data" / "processed" / "canonical").rglob("*.txt"):
        if label not in known_labels:
            invalid.append({"sample_id": "", "image_path": "", "label_path": str(label), "severity": "warning", "reason": "orphan_label"})
    write_csv_dicts(MANIFEST_DIR / "invalid_annotations.csv", invalid, FIELDS)
    severe = sum(1 for row in invalid if row["severity"] == "severe")
    md = [
        "# Annotation Validation Report",
        "",
        f"- Samples checked: `{len(rows)}`",
        f"- Invalid rows: `{len(invalid)}`",
        f"- Severe unresolved errors: `{severe}`",
        "",
        "Training is blocked when severe unresolved errors are present.",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "annotation_validation_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 1 if severe else 0


if __name__ == "__main__":
    raise SystemExit(main())
