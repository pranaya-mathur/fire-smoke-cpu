#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_smoke_cpu.constants import MANIFEST_DIR, REPORT_DIR
from fire_smoke_cpu.utils import write_csv_dicts


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "all_samples.csv")
    parser.add_argument("--near-threshold", type=int, default=6)
    args = parser.parse_args()
    if not args.manifest.exists():
        print(f"Missing manifest: {args.manifest}")
        return 2
    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8")))
    by_sha = defaultdict(list)
    for row in rows:
        by_sha[row["sha256"]].append(row)
    exact_rows = []
    for group_id, (_sha, group) in enumerate((item for item in by_sha.items() if len(item[1]) > 1), start=1):
        sha, members = _sha, group
        keep = members[0]["sample_id"]
        for member in members:
            exact_rows.append({"duplicate_group": group_id, "sha256": sha, "sample_id": member["sample_id"], "keep_sample_id": keep, "split": member.get("split", "")})
    near_rows = []
    for group_id, (left, right) in enumerate(itertools.combinations(rows, 2), start=1):
        if not left.get("perceptual_hash") or not right.get("perceptual_hash"):
            continue
        try:
            distance = hamming_hex(left["perceptual_hash"], right["perceptual_hash"])
        except ValueError:
            continue
        if 0 < distance <= args.near_threshold:
            near_rows.append(
                {
                    "near_group": group_id,
                    "sample_id_a": left["sample_id"],
                    "sample_id_b": right["sample_id"],
                    "phash_distance": distance,
                    "split_a": left.get("split", ""),
                    "split_b": right.get("split", ""),
                }
            )
    write_csv_dicts(MANIFEST_DIR / "exact_duplicates.csv", exact_rows, ["duplicate_group", "sha256", "sample_id", "keep_sample_id", "split"])
    write_csv_dicts(MANIFEST_DIR / "near_duplicates.csv", near_rows, ["near_group", "sample_id_a", "sample_id_b", "phash_distance", "split_a", "split_b"])
    cross_split_exact = [r for r in exact_rows if r.get("split") and r["split"] != next((k["split"] for k in rows if k["sample_id"] == r["keep_sample_id"]), r["split"])]
    md = [
        "# Deduplication Report",
        "",
        f"- Samples checked: `{len(rows)}`",
        f"- Exact duplicate member rows: `{len(exact_rows)}`",
        f"- Near-duplicate pairs at threshold {args.near_threshold}: `{len(near_rows)}`",
        f"- Exact duplicate cross-split issues: `{len(cross_split_exact)}`",
        "",
        "Policy: exact duplicate copies may be excluded after retaining one canonical sample; near duplicates are grouped for split co-location and human review.",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "deduplication_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 1 if cross_split_exact else 0


if __name__ == "__main__":
    raise SystemExit(main())
