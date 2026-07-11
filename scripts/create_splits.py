#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_smoke_cpu.constants import CLASS_ID_TO_NAME, MANIFEST_COLUMNS, MANIFEST_DIR, PROCESSED_DIR, REPORT_DIR
from fire_smoke_cpu.utils import safe_link_or_copy, write_csv_dicts


def group_key(row: dict) -> str:
    return row.get("group_id") or row["sha256"] or row["sample_id"]


def label_for(row: dict) -> str:
    fire = row.get("has_fire") in {"True", "true", "1", "yes"}
    smoke = row.get("has_smoke") in {"True", "true", "1", "yes"}
    if fire and smoke:
        return "fire_smoke"
    if fire:
        return "fire_only"
    if smoke:
        return "smoke_only"
    return "negative"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "all_samples.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.manifest.exists():
        print(f"Missing manifest: {args.manifest}")
        return 2
    rows = [r for r in csv.DictReader(args.manifest.open("r", encoding="utf-8")) if r.get("excluded") not in {"True", "true", "1"}]
    if not rows:
        print("No eligible rows for splitting.")
        return 2
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    group_items = list(groups.items())
    random.Random(args.seed).shuffle(group_items)
    split_targets = {"train": 0.70, "val": 0.15, "test": 0.15}
    split_rows = {"train": [], "val": [], "test": []}
    target_counts = {name: int(len(rows) * frac) for name, frac in split_targets.items()}
    for _gid, members in group_items:
        split = min(split_rows.keys(), key=lambda name: len(split_rows[name]) / max(target_counts[name], 1))
        split_rows[split].extend(members)
    out_root = PROCESSED_DIR / "fire_smoke_v1"
    for split, members in split_rows.items():
        for row in members:
            row["split"] = split
            img_src = Path(row["canonical_image_path"])
            lbl_src = Path(row["canonical_label_path"])
            img_dst = out_root / "images" / split / img_src.name
            lbl_dst = out_root / "labels" / split / lbl_src.name
            safe_link_or_copy(img_src, img_dst)
            safe_link_or_copy(lbl_src, lbl_dst)
    yaml_path = out_root / "fire_smoke.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: fire",
                "  1: smoke",
                "",
            ]
        ),
        encoding="utf-8",
    )
    all_rows = [row for split in ["train", "val", "test"] for row in split_rows[split]]
    write_csv_dicts(args.manifest, all_rows, MANIFEST_COLUMNS)
    exact_leaks = 0
    sha_to_splits = defaultdict(set)
    for row in all_rows:
        sha_to_splits[row["sha256"]].add(row["split"])
    exact_leaks = sum(1 for splits in sha_to_splits.values() if len(splits) > 1)
    counts = {split: len(members) for split, members in split_rows.items()}
    label_counts = {split: Counter(label_for(row) for row in members) for split, members in split_rows.items()}
    source_counts = {split: Counter(row["source_dataset"] for row in members) for split, members in split_rows.items()}
    md = [
        "# Split Report",
        "",
        f"- Total samples: `{len(all_rows)}`",
        f"- Train/val/test counts: `{counts}`",
        f"- Label counts: `{dict(label_counts)}`",
        f"- Source counts: `{dict(source_counts)}`",
        f"- Exact duplicate split leakage checks: `{exact_leaks}`",
        f"- Seed: `{args.seed}`",
        "",
        "Splits are group-aware using inferred source/video/filename groups. Exact and near-duplicate reports should be reviewed before final training.",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "split_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 1 if exact_leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
