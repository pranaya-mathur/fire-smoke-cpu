#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_smoke_cpu.annotations import parse_yolo_label
from fire_smoke_cpu.constants import CLASS_ID_TO_NAME, MANIFEST_DIR, REPORT_DIR
from fire_smoke_cpu.utils import ensure_dirs, open_image


def category(row: dict) -> str:
    fire = row.get("has_fire") in {"True", "true", "1"}
    smoke = row.get("has_smoke") in {"True", "true", "1"}
    if fire and smoke:
        return "fire_and_smoke"
    if fire:
        return "fire_only"
    if smoke:
        return "smoke_only"
    return "negative"


def render(row: dict, out: Path) -> None:
    from PIL import ImageDraw, ImageFont

    with open_image(Path(row["canonical_image_path"])).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        boxes, _errors = parse_yolo_label(Path(row["canonical_label_path"]))
        for box in boxes:
            x1 = (box.x_center - box.width / 2) * image.width
            y1 = (box.y_center - box.height / 2) * image.height
            x2 = (box.x_center + box.width / 2) * image.width
            y2 = (box.y_center + box.height / 2) * image.height
            color = (255, 64, 32) if box.class_id == 0 else (80, 160, 255)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, image.width // 300))
            draw.text((x1, max(0, y1 - 14)), CLASS_ID_TO_NAME[box.class_id], fill=color)
        image.thumbnail((640, 640))
        image.save(out, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "all_samples.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.manifest.exists():
        print(f"Missing manifest: {args.manifest}")
        return 2
    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8")))
    out_dir = REPORT_DIR / "visual_qc"
    ensure_dirs(out_dir)
    rng = random.Random(args.seed)
    limits = {"fire_only": 50, "smoke_only": 50, "fire_and_smoke": 50, "negative": 100}
    index_lines = ["# Visual QC Index", ""]
    for cat, limit in limits.items():
        candidates = [row for row in rows if category(row) == cat]
        rng.shuffle(candidates)
        index_lines.append(f"## {cat}")
        for i, row in enumerate(candidates[:limit], start=1):
            out = out_dir / f"{cat}_{i:03d}_{row['sample_id']}.jpg"
            render(row, out)
            index_lines.append(f"- [{out.name}]({out.name})")
        index_lines.append("")
    (out_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"Visual QC written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
