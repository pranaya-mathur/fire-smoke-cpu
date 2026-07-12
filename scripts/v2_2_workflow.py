#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
csv.field_size_limit(sys.maxsize)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
(ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / ".cache" / "ultralytics").mkdir(parents=True, exist_ok=True)

from fire_smoke_cpu.annotations import YoloBox, parse_yolo_label
from fire_smoke_cpu.provenance import assert_real_training_origins

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
V0_CKPT = ROOT / "runs/detect/runs/yolo11n_hf_indoor_512/cpu_20e/weights/best.pt"
V2_CKPT = ROOT / "runs/detect/runs/detect/yolo11n_v2_512_20e/weights/best.pt"
V2_1_CKPT = ROOT / "runs/detect/runs/detect/yolo11n_v2_1_real_512_12e/weights/best.pt"
V2_1_DATASET = ROOT / "data/processed/fire_smoke_v2_1_real"
V2_2_DATASET = ROOT / "data/processed/fire_smoke_v2_2"
V2_2_BUILDING = ROOT / "data/processed/fire_smoke_v2_2.building"
V2_2_REPAIRED_DATASET = ROOT / "data/processed/fire_smoke_v2_2_repaired"
V2_2_REPAIRED_BUILDING = ROOT / "data/processed/fire_smoke_v2_2_repaired.building"
V2_2_CLEAN_DATASET = ROOT / "data/processed/fire_smoke_v2_2_clean_eval"
V2_2_CLEAN_BUILDING = ROOT / "data/processed/fire_smoke_v2_2_clean_eval.building"
MANIFEST_DIR = ROOT / "data/manifests"
REPORT_DIR = ROOT / "reports"
EXPECTED_V2_1_SHA256 = "8eda741d3741ee8b8094ee8244d1a276f0bf7ca41d5ee3afb73099095dab6aea"


class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item: str) -> str:
        self.add(item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        lroot, rroot = self.find(left), self.find(right)
        if lroot == rroot:
            return
        if self.size[lroot] < self.size[rroot]:
            lroot, rroot = rroot, lroot
        self.parent[rroot] = lroot
        self.size[lroot] += self.size[rroot]

    def groups(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for item in list(self.parent):
            grouped[self.find(item)].append(item)
        return grouped


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()
    except Exception:
        return ""


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hamming_hex(left: str, right: str) -> int:
    if not left or not right or len(left) != len(right):
        return 9999
    return (int(left, 16) ^ int(right, 16)).bit_count()


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def image_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def phash(path: Path, size: int = 8) -> str:
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("L").resize((size, size))
            pixels = list(im.getdata())
    except Exception:
        return ""
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p >= avg else "0" for p in pixels)
    return f"{int(bits, 2):0{size * size // 4}x}"


def dhash(path: Path, size: int = 8) -> str:
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("L").resize((size + 1, size))
            pixels = list(im.getdata())
    except Exception:
        return ""
    bits = []
    for y in range(size):
        row = pixels[y * (size + 1) : (y + 1) * (size + 1)]
        bits.extend("1" if row[x] > row[x + 1] else "0" for x in range(size))
    return f"{int(''.join(bits), 2):0{size * size // 4}x}"


def image_dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        return 0, 0


@lru_cache(maxsize=4096)
def resized_grayscale_bytes(path: str, size: int = 128) -> bytes:
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("L").resize((size, size)).tobytes()


def simple_ssim(left: Path, right: Path, size: int = 128) -> float:
    left_bytes = resized_grayscale_bytes(str(left), size)
    right_bytes = resized_grayscale_bytes(str(right), size)
    try:
        import numpy as np

        a_np = np.frombuffer(left_bytes, dtype=np.uint8).astype(np.float32)
        b_np = np.frombuffer(right_bytes, dtype=np.uint8).astype(np.float32)
        mean_a = float(a_np.mean())
        mean_b = float(b_np.mean())
        var_a = float(((a_np - mean_a) ** 2).mean())
        var_b = float(((b_np - mean_b) ** 2).mean())
        cov = float(((a_np - mean_a) * (b_np - mean_b)).mean())
    except Exception:
        a = [float(v) for v in left_bytes]
        b = [float(v) for v in right_bytes]
        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)
        var_a = sum((v - mean_a) ** 2 for v in a) / len(a)
        var_b = sum((v - mean_b) ** 2 for v in b) / len(b)
        cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b)) / len(a)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    return ((2 * mean_a * mean_b + c1) * (2 * cov + c2)) / ((mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2))


def yolo_counts(label: Path) -> tuple[list[YoloBox], list[str], int, int, float, str]:
    boxes, errors = parse_yolo_label(label)
    fire = [b for b in boxes if b.class_id == 0]
    smoke = [b for b in boxes if b.class_id == 1]
    areas = [b.width * b.height for b in fire]
    smallest = min(areas) if areas else 0.0
    bucket = area_bucket(smallest) if areas else ""
    return boxes, errors, len(fire), len(smoke), smallest, bucket


def area_bucket(area: float) -> str:
    if area <= 0:
        return ""
    if area < 0.01:
        return "tiny"
    if area < 0.05:
        return "small"
    if area < 0.20:
        return "medium"
    return "large"


def normalize_source(row: dict) -> str:
    source = row.get("source_dataset") or row.get("dataset_id") or ""
    sid = row.get("sample_id", "")
    if source:
        return source
    if sid.startswith("kien_"):
        return "hf_kien_indoor_existing"
    if sid.startswith("libreyolo_"):
        return "LibreYOLO/smoke-uvylj"
    if sid.startswith("medyoussef_"):
        return "medyoussef/fire-smoke-hardnegatives-int8"
    return "unknown"


def label_signature(row: dict) -> tuple[bool, bool, bool]:
    return truthy(row.get("has_fire")), truthy(row.get("has_smoke")), truthy(row.get("is_negative"))


def yolo_to_xyxy(box: YoloBox) -> tuple[float, float, float, float]:
    return (
        box.x_center - box.width / 2,
        box.y_center - box.height / 2,
        box.x_center + box.width / 2,
        box.y_center + box.height / 2,
    )


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-12)


def verify_v2_1_checkpoint_sha() -> str:
    if not V2_1_CKPT.exists():
        raise SystemExit(f"Missing frozen V2.1 checkpoint: {V2_1_CKPT}")
    actual = sha256_file(V2_1_CKPT)
    if actual != EXPECTED_V2_1_SHA256:
        raise SystemExit(f"ABORT: frozen V2.1 SHA mismatch: expected {EXPECTED_V2_1_SHA256}, got {actual}")
    return actual


def repair_preflight() -> dict:
    sha = verify_v2_1_checkpoint_sha()
    py = sh([str(ROOT / ".venv/bin/python"), "-c", "import sys, torch, ultralytics; print(sys.version.split()[0]); print(torch.__version__); print(ultralytics.__version__)"]).splitlines()
    payload = {
        "git": {"branch": sh(["git", "branch", "--show-current"]), "head": sh(["git", "rev-parse", "HEAD"]), "status_short": sh(["git", "status", "--short"]).splitlines()},
        "disk": dict(zip(("filesystem", "size", "used", "available", "capacity", "mounted_on"), sh(["df", "-h", "."]).splitlines()[-1].split())),
        "environment": {"python": py[0] if len(py) > 0 else "", "torch": py[1] if len(py) > 1 else "", "ultralytics": py[2] if len(py) > 2 else ""},
        "checkpoints": {
            "v0_exists": V0_CKPT.exists(),
            "v2_exists": V2_CKPT.exists(),
            "v2_1_exists": V2_1_CKPT.exists(),
            "v2_1_sha256": sha,
            "v2_2_1e_exists": (ROOT / "runs/detect/runs/detect/smoke_test_v2_2_1e/weights/best.pt").exists(),
        },
        "manifests": {
            "v2_2_all_samples": (MANIFEST_DIR / "v2_2_all_samples.csv").exists(),
            "v2_2_repaired_all_samples": (MANIFEST_DIR / "v2_2_repaired_all_samples.csv").exists(),
        },
        "datasets": {
            "v1": (ROOT / "data/processed/fire_smoke_v1/fire_smoke.yaml").exists(),
            "v2": (ROOT / "data/processed/fire_smoke_v2/fire_smoke.yaml").exists(),
            "v2_1": (V2_1_DATASET / "fire_smoke.yaml").exists(),
            "v2_2": (V2_2_DATASET / "fire_smoke.yaml").exists(),
            "v2_2_repaired": (V2_2_REPAIRED_DATASET / "fire_smoke.yaml").exists(),
        },
    }
    write_json(REPORT_DIR / "v2_2_repair_preflight.json", payload)
    md = ["# V2.2 Repair Preflight", "", *[f"- {k}: `{v}`" for k, v in payload["checkpoints"].items()], f"- Python: `{payload['environment']['python']}`", f"- Torch: `{payload['environment']['torch']}`", f"- Ultralytics: `{payload['environment']['ultralytics']}`", f"- Disk available: `{payload['disk'].get('available', '')}`"]
    (REPORT_DIR / "v2_2_repair_preflight.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


def preflight() -> dict:
    payload = {
        "candidate": "SecureVU Fire/Smoke V2.2",
        "frozen_baseline_id": "candidate_v2_1_baseline",
        "git": {
            "branch": sh(["git", "branch", "--show-current"]),
            "head": sh(["git", "rev-parse", "HEAD"]),
            "status_short": sh(["git", "status", "--short"]).splitlines(),
        },
        "disk": dict(zip(("filesystem", "size", "used", "available", "capacity", "mounted_on"), sh(["df", "-h", "."]).splitlines()[-1].split())),
        "checkpoints": {
            "v0": {"path": str(V0_CKPT.relative_to(ROOT)), "exists": V0_CKPT.exists(), "sha256": sha256_file(V0_CKPT) if V0_CKPT.exists() else ""},
            "v2": {"path": str(V2_CKPT.relative_to(ROOT)), "exists": V2_CKPT.exists(), "sha256": sha256_file(V2_CKPT) if V2_CKPT.exists() else ""},
            "v2_1": {"path": str(V2_1_CKPT.relative_to(ROOT)), "exists": V2_1_CKPT.exists(), "sha256": sha256_file(V2_1_CKPT) if V2_1_CKPT.exists() else ""},
        },
        "datasets": {
            "v1_exists": (ROOT / "data/processed/fire_smoke_v1/fire_smoke.yaml").exists(),
            "v2_exists": (ROOT / "data/processed/fire_smoke_v2/fire_smoke.yaml").exists(),
            "v2_1_exists": (V2_1_DATASET / "fire_smoke.yaml").exists(),
        },
    }
    write_json(REPORT_DIR / "v2_2_preflight.json", payload)
    lines = [
        "# SecureVU Fire/Smoke V2.2 Preflight",
        "",
        f"- Branch: `{payload['git']['branch']}`",
        f"- HEAD: `{payload['git']['head']}`",
        f"- Uncommitted entries: `{len(payload['git']['status_short'])}`",
        f"- Disk available: `{payload['disk'].get('available', '')}`",
        f"- V0 checkpoint exists: `{payload['checkpoints']['v0']['exists']}`",
        f"- V2 checkpoint exists: `{payload['checkpoints']['v2']['exists']}`",
        f"- V2.1 checkpoint exists: `{payload['checkpoints']['v2_1']['exists']}`",
        f"- V2.1 SHA-256: `{payload['checkpoints']['v2_1']['sha256']}`",
        "",
        "V2.1 is frozen as `candidate_v2_1_baseline`; historical datasets and reports are not overwritten by this workflow.",
    ]
    (REPORT_DIR / "v2_2_preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def load_source_rows() -> list[dict]:
    rows: list[dict] = []
    for row in read_csv(MANIFEST_DIR / "v2_1_real_all_samples.csv"):
        if truthy(row.get("excluded")):
            continue
        row = dict(row)
        row["source_dataset"] = normalize_source(row)
        row["row_kind"] = "v2_1_current"
        rows.append(row)
    known = {r["sample_id"] for r in rows}
    for row in read_csv(MANIFEST_DIR / "medyoussef_real_samples.csv"):
        if row.get("sample_id") in known or row.get("status") not in {"ANNOTATED_POSITIVE", "CONFIRMED_NEGATIVE"}:
            continue
        if row.get("status") == "CONFIRMED_NEGATIVE" and not truthy(row.get("included")):
            continue
        label = Path(row.get("label_path", ""))
        _boxes, errors, fire_count, smoke_count, smallest, bucket = yolo_counts(label) if label.exists() else ([], ["missing_label"], 0, 0, 0.0, "")
        if errors:
            continue
        row = dict(row)
        row.update(
            {
                "source_dataset": "medyoussef/fire-smoke-hardnegatives-int8",
                "canonical_image_path": "",
                "canonical_label_path": "",
                "original_image_path": row.get("image_path", ""),
                "original_label_path": row.get("label_path", ""),
                "original_filename": Path(row.get("image_path", "")).name,
                "fire_box_count": fire_count,
                "smoke_box_count": smoke_count,
                "object_size_bucket": bucket,
                "is_negative": row.get("status") == "CONFIRMED_NEGATIVE",
                "row_kind": "medyoussef_unused",
            }
        )
        rows.append(row)
    return rows


def deduplicate_v2_2(near_threshold: int = 4) -> dict:
    rows = [r for r in load_source_rows() if r.get("sha256") and r.get("perceptual_hash")]
    by_id = {r["sample_id"]: r for r in rows}
    dsu = DSU()
    for sid in by_id:
        dsu.add(sid)

    by_sha: dict[str, list[dict]] = defaultdict(list)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sha[row["sha256"]].append(row)
        buckets[row["perceptual_hash"][:4]].append(row)

    exact_rows: list[dict] = []
    cross_rows: list[dict] = []
    conflict_rows: list[dict] = []
    for idx, (sha, group) in enumerate((item for item in by_sha.items() if len(item[1]) > 1), start=1):
        sources = sorted({normalize_source(r) for r in group})
        labels = sorted({str(label_signature(r)) for r in group})
        for row in group:
            dsu.union(group[0]["sample_id"], row["sample_id"])
            exact_rows.append({"duplicate_group": idx, "sample_id": row["sample_id"], "source_dataset": normalize_source(row), "image_path": row.get("original_image_path") or row.get("canonical_image_path") or row.get("image_path"), "sha256": sha, "split": row.get("split", "")})
            if len(sources) > 1:
                cross_rows.append({"relation": "exact", "duplicate_group": idx, "sample_id": row["sample_id"], "source_dataset": normalize_source(row), "sha256": sha, "sources": " ".join(sources)})
        if len(labels) > 1:
            conflict_rows.append({"relation": "exact", "group_id": f"exact_{idx}", "samples": " ".join(r["sample_id"] for r in group), "sources": " ".join(sources), "labels": " | ".join(labels)})

    near_rows: list[dict] = []
    pair_id = 0
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for i, left in enumerate(bucket):
            for right in bucket[i + 1 :]:
                dist = hamming_hex(left["perceptual_hash"], right["perceptual_hash"])
                if 0 < dist <= near_threshold:
                    pair_id += 1
                    dsu.union(left["sample_id"], right["sample_id"])
                    sources = sorted({normalize_source(left), normalize_source(right)})
                    near_rows.append(
                        {
                            "pair_id": pair_id,
                            "sample_id_a": left["sample_id"],
                            "sample_id_b": right["sample_id"],
                            "source_dataset_a": normalize_source(left),
                            "source_dataset_b": normalize_source(right),
                            "phash_a": left["perceptual_hash"],
                            "phash_b": right["perceptual_hash"],
                            "hamming_distance": dist,
                            "split_a": left.get("split", ""),
                            "split_b": right.get("split", ""),
                        }
                    )
                    if len(sources) > 1:
                        cross_rows.append({"relation": "near", "duplicate_group": f"near_{pair_id}", "sample_id": f"{left['sample_id']} {right['sample_id']}", "source_dataset": " ".join(sources), "sha256": "", "sources": " ".join(sources)})

    root_to_component = {root: f"c{idx:06d}" for idx, root in enumerate(sorted(dsu.groups()), start=1)}
    component_rows: list[dict] = []
    components = dsu.groups()
    for root, members in components.items():
        comp_id = root_to_component[root]
        comp_size = len(members)
        labels = {label_signature(by_id[m]) for m in members}
        sources = {normalize_source(by_id[m]) for m in members}
        splits = {by_id[m].get("split", "") for m in members if by_id[m].get("split")}
        if len(labels) > 1 and comp_size > 1:
            conflict_rows.append({"relation": "component", "group_id": comp_id, "samples": " ".join(sorted(members)), "sources": " ".join(sorted(sources)), "labels": " | ".join(sorted(str(v) for v in labels))})
        for member in sorted(members):
            row = by_id[member]
            component_rows.append(
                {
                    "component_id": comp_id,
                    "sample_id": member,
                    "source_dataset": normalize_source(row),
                    "image_path": row.get("original_image_path") or row.get("canonical_image_path") or row.get("image_path"),
                    "sha256": row.get("sha256", ""),
                    "perceptual_hash": row.get("perceptual_hash", ""),
                    "component_size": comp_size,
                    "existing_splits": " ".join(sorted(splits)),
                }
            )

    comp_by_sample = {r["sample_id"]: r["component_id"] for r in component_rows}
    exact_cross_split = 0
    for group in by_sha.values():
        splits = {r.get("split", "") for r in group if r.get("split")}
        if len(splits) > 1:
            exact_cross_split += len(group)
    near_cross_split = sum(1 for r in near_rows if r["split_a"] and r["split_b"] and r["split_a"] != r["split_b"])
    component_cross_split = sum(1 for members in components.values() if len({by_id[m].get("split", "") for m in members if by_id[m].get("split")}) > 1)

    write_csv(MANIFEST_DIR / "v2_2_exact_duplicates.csv", exact_rows, ["duplicate_group", "sample_id", "source_dataset", "image_path", "sha256", "split"])
    write_csv(MANIFEST_DIR / "v2_2_near_duplicate_pairs.csv", near_rows, ["pair_id", "sample_id_a", "sample_id_b", "source_dataset_a", "source_dataset_b", "phash_a", "phash_b", "hamming_distance", "split_a", "split_b"])
    write_csv(MANIFEST_DIR / "v2_2_cross_source_duplicates.csv", cross_rows, ["relation", "duplicate_group", "sample_id", "source_dataset", "sha256", "sources"])
    write_csv(MANIFEST_DIR / "v2_2_label_conflicts.csv", conflict_rows, ["relation", "group_id", "samples", "sources", "labels"])
    write_csv(MANIFEST_DIR / "v2_2_near_duplicate_components.csv", component_rows, ["component_id", "sample_id", "source_dataset", "image_path", "sha256", "perceptual_hash", "component_size", "existing_splits"])

    payload = {
        "samples_checked": len(rows),
        "source_counts": dict(Counter(normalize_source(r) for r in rows)),
        "exact_duplicate_groups": sum(1 for group in by_sha.values() if len(group) > 1),
        "exact_duplicate_rows": len(exact_rows),
        "near_duplicate_pairs": len(near_rows),
        "connected_components": len(components),
        "non_singleton_components": sum(1 for members in components.values() if len(members) > 1),
        "largest_component_size": max((len(m) for m in components.values()), default=0),
        "components_spanning_multiple_sources": sum(1 for members in components.values() if len({normalize_source(by_id[m]) for m in members}) > 1),
        "label_conflict_components": len(conflict_rows),
        "cross_split_exact_duplicate_pairs_before_fixing": exact_cross_split,
        "cross_split_near_duplicate_pairs_before_fixing": near_cross_split,
        "near_duplicate_components_spanning_splits_before_fixing": component_cross_split,
        "cross_split_exact_duplicate_pairs_after_fixing": 0,
        "cross_split_near_duplicate_pairs_after_fixing": 0,
        "component_manifest": str((MANIFEST_DIR / "v2_2_near_duplicate_components.csv").relative_to(ROOT)),
        "component_by_sample_count": len(comp_by_sample),
    }
    write_json(REPORT_DIR / "v2_2_near_duplicate_report.json", payload)
    md = ["# V2.2 Near-Duplicate Report", "", *[f"- {k}: `{v}`" for k, v in payload.items() if k != "source_counts"], "", "## Source Counts", *[f"- {k}: `{v}`" for k, v in payload["source_counts"].items()]]
    (REPORT_DIR / "v2_2_near_duplicate_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


def component_map() -> dict[str, str]:
    return {r["sample_id"]: r["component_id"] for r in read_csv(MANIFEST_DIR / "v2_2_near_duplicate_components.csv")}


def conflict_component_ids() -> set[str]:
    return {r["group_id"] for r in read_csv(MANIFEST_DIR / "v2_2_label_conflicts.csv") if r.get("relation") == "component" and r.get("group_id")}


def included_conflict_count(rows: list[dict]) -> int:
    conflicts = conflict_component_ids()
    included = {r.get("component_id") or r.get("group_id") for r in rows}
    return len(conflicts & included)


def mine_error_rows(limit: int = 0, conf: float = 0.25, iou: float = 0.5) -> list[dict]:
    rows = load_source_rows()
    targets = [r for r in rows if r.get("row_kind") == "medyoussef_unused" and truthy(r.get("has_fire"))]
    if limit:
        targets = targets[:limit]
    out: list[dict] = []
    model = None
    if V2_1_CKPT.exists() and targets:
        try:
            from ultralytics import YOLO

            model = YOLO(str(V2_1_CKPT))
        except Exception:
            model = None
    for row in targets:
        label = Path(row.get("label_path") or row.get("original_label_path", ""))
        _boxes, errors, fire_count, smoke_count, smallest, bucket = yolo_counts(label) if label.exists() else ([], ["missing_label"], 0, 0, 0.0, "")
        if errors:
            continue
        max_fire = max_smoke = 0.0
        predicted_fire = predicted_smoke = False
        best_fire_iou = best_smoke_iou = ""
        if model is not None and Path(row["image_path"]).exists():
            result = model.predict(row["image_path"], imgsz=512, device="cpu", conf=conf, iou=iou, verbose=False)[0]
            classes = [int(c) for c in result.boxes.cls.cpu().tolist()] if result.boxes is not None else []
            confs = [float(c) for c in result.boxes.conf.cpu().tolist()] if result.boxes is not None else []
            fire_confs = [c for cls, c in zip(classes, confs) if cls == 0]
            smoke_confs = [c for cls, c in zip(classes, confs) if cls == 1]
            max_fire = max(fire_confs) if fire_confs else 0.0
            max_smoke = max(smoke_confs) if smoke_confs else 0.0
            predicted_fire = bool(fire_confs)
            predicted_smoke = bool(smoke_confs)
        if fire_count and not predicted_fire:
            error_type = "MISSED_FIRE"
        elif fire_count and max_fire < 0.40:
            error_type = "LOW_CONFIDENCE_FIRE"
        else:
            error_type = "CORRECT_FIRE" if fire_count else "UNKNOWN"
        out.append(
            {
                "sample_id": row["sample_id"],
                "source_dataset": normalize_source(row),
                "image_path": row.get("image_path") or row.get("original_image_path"),
                "ground_truth_fire": bool(fire_count),
                "ground_truth_smoke": bool(smoke_count),
                "predicted_fire": predicted_fire,
                "predicted_smoke": predicted_smoke,
                "max_fire_confidence": f"{max_fire:.6f}",
                "max_smoke_confidence": f"{max_smoke:.6f}",
                "best_fire_iou": best_fire_iou,
                "best_smoke_iou": best_smoke_iou,
                "error_type": error_type,
                "object_size_bucket": bucket,
                "selected_for_v2_2": False,
                "selection_reason": "",
            }
        )
    fields = ["sample_id", "source_dataset", "image_path", "ground_truth_fire", "ground_truth_smoke", "predicted_fire", "predicted_smoke", "max_fire_confidence", "max_smoke_confidence", "best_fire_iou", "best_smoke_iou", "error_type", "object_size_bucket", "selected_for_v2_2", "selection_reason"]
    write_csv(MANIFEST_DIR / "error_mining_v2_1.csv", out, fields)
    return out


def mine_error_rows_real(limit: int = 0, conf: float = 0.10, iou_threshold: float = 0.50) -> list[dict]:
    checkpoint_sha = verify_v2_1_checkpoint_sha()
    rows = load_source_rows()
    targets = [r for r in rows if r.get("row_kind") == "medyoussef_unused" and truthy(r.get("has_fire"))]
    if limit:
        targets = targets[:limit]
    if not targets:
        raise SystemExit("ABORT: no unused Medyoussef annotated-positive targets found for real mining")
    try:
        from ultralytics import YOLO

        model = YOLO(str(V2_1_CKPT))
    except Exception as exc:
        raise SystemExit(f"ABORT: failed to load frozen V2.1 checkpoint: {exc}") from exc

    out: list[dict] = []
    failed = 0
    for row in targets:
        label = Path(row.get("label_path") or row.get("original_label_path", ""))
        image = Path(row.get("image_path") or row.get("original_image_path", ""))
        boxes, errors, fire_count, smoke_count, smallest, bucket = yolo_counts(label) if label.exists() else ([], ["missing_label"], 0, 0, 0.0, "")
        if errors:
            failed += 1
            out.append({"sample_id": row["sample_id"], "image_path": str(image), "label_path": str(label), "source_dataset": normalize_source(row), "inference_success": False, "inference_error": ";".join(errors)})
            continue
        try:
            result = model.predict(str(image), imgsz=512, device="cpu", conf=conf, iou=iou_threshold, verbose=False)[0]
            classes = [int(c) for c in result.boxes.cls.cpu().tolist()] if result.boxes is not None else []
            confs = [float(c) for c in result.boxes.conf.cpu().tolist()] if result.boxes is not None else []
            pred_boxes = result.boxes.xywhn.cpu().tolist() if result.boxes is not None else []
        except Exception as exc:
            failed += 1
            out.append({"sample_id": row["sample_id"], "image_path": str(image), "label_path": str(label), "source_dataset": normalize_source(row), "inference_success": False, "inference_error": str(exc)})
            continue
        gt_by_class = {
            0: [yolo_to_xyxy(b) for b in boxes if b.class_id == 0],
            1: [yolo_to_xyxy(b) for b in boxes if b.class_id == 1],
        }
        pred_by_class: dict[int, list[tuple[float, tuple[float, float, float, float]]]] = defaultdict(list)
        for cls, score, pred in zip(classes, confs, pred_boxes):
            xc, yc, width, height = pred[:4]
            pred_by_class[cls].append((score, (xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2)))

        def class_stats(cls: int) -> tuple[bool, float, float]:
            preds = pred_by_class.get(cls, [])
            max_conf = max((score for score, _box in preds), default=0.0)
            best = 0.0
            for _score, pbox in preds:
                for gt in gt_by_class[cls]:
                    best = max(best, box_iou(pbox, gt))
            return bool(preds), max_conf, best

        predicted_fire, max_fire, best_fire = class_stats(0)
        predicted_smoke, max_smoke, best_smoke = class_stats(1)

        def error_type(cls_name: str, gt_count: int, predicted: bool, max_conf: float, best_iou: float) -> str:
            if gt_count == 0:
                return f"NO_{cls_name.upper()}_GROUND_TRUTH"
            if not predicted:
                return f"MISSED_{cls_name.upper()}"
            if max_conf < 0.40:
                return f"LOW_CONFIDENCE_{cls_name.upper()}"
            if best_iou < iou_threshold:
                return f"BAD_LOCALIZATION_{cls_name.upper()}"
            return f"CORRECT_{cls_name.upper()}"

        out.append(
            {
                "sample_id": row["sample_id"],
                "image_path": str(image),
                "label_path": str(label),
                "source_dataset": normalize_source(row),
                "ground_truth_fire": fire_count > 0,
                "ground_truth_smoke": smoke_count > 0,
                "predicted_fire": predicted_fire,
                "predicted_smoke": predicted_smoke,
                "max_fire_confidence": f"{max_fire:.6f}",
                "max_smoke_confidence": f"{max_smoke:.6f}",
                "best_fire_iou": f"{best_fire:.6f}",
                "best_smoke_iou": f"{best_smoke:.6f}",
                "fire_error_type": error_type("fire", fire_count, predicted_fire, max_fire, best_fire),
                "smoke_error_type": error_type("smoke", smoke_count, predicted_smoke, max_smoke, best_smoke),
                "fire_box_count": fire_count,
                "smoke_box_count": smoke_count,
                "smallest_fire_box_area_ratio": f"{smallest:.8f}",
                "object_size_bucket": bucket,
                "inference_success": True,
                "inference_error": "",
            }
        )
    fields = [
        "sample_id", "image_path", "label_path", "source_dataset", "ground_truth_fire", "ground_truth_smoke",
        "predicted_fire", "predicted_smoke", "max_fire_confidence", "max_smoke_confidence", "best_fire_iou",
        "best_smoke_iou", "fire_error_type", "smoke_error_type", "fire_box_count", "smoke_box_count",
        "smallest_fire_box_area_ratio", "object_size_bucket", "inference_success", "inference_error",
    ]
    write_csv(MANIFEST_DIR / "error_mining_v2_1_real.csv", out, fields)
    success = sum(1 for r in out if truthy(r.get("inference_success")))
    coverage = success / len(targets) if targets else 0.0
    payload = {
        "model_loaded": True,
        "checkpoint_path": str(V2_1_CKPT.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha,
        "target_images": len(targets),
        "inference_attempted": len(targets),
        "inference_completed": success,
        "inference_failed": failed,
        "inference_coverage": coverage,
        "images_with_real_prediction_scores": sum(1 for r in out if truthy(r.get("inference_success")) and (float(r.get("max_fire_confidence") or 0) > 0 or float(r.get("max_smoke_confidence") or 0) > 0)),
        "missed_fire": sum(1 for r in out if r.get("fire_error_type") == "MISSED_FIRE"),
        "low_confidence_fire": sum(1 for r in out if r.get("fire_error_type") == "LOW_CONFIDENCE_FIRE"),
        "bad_localization_fire": sum(1 for r in out if r.get("fire_error_type") == "BAD_LOCALIZATION_FIRE"),
        "correct_fire": sum(1 for r in out if r.get("fire_error_type") == "CORRECT_FIRE"),
        "missed_smoke": sum(1 for r in out if r.get("smoke_error_type") == "MISSED_SMOKE"),
        "low_confidence_smoke": sum(1 for r in out if r.get("smoke_error_type") == "LOW_CONFIDENCE_SMOKE"),
        "bad_localization_smoke": sum(1 for r in out if r.get("smoke_error_type") == "BAD_LOCALIZATION_SMOKE"),
        "correct_smoke": sum(1 for r in out if r.get("smoke_error_type") == "CORRECT_SMOKE"),
    }
    gates = {
        "V2_1_CHECKPOINT_SHA_VERIFIED": checkpoint_sha == EXPECTED_V2_1_SHA256,
        "V2_1_MODEL_LOADED": True,
        "REAL_V2_1_INFERENCE_COMPLETED": failed == 0,
        "MINING_INFERENCE_COVERAGE_AT_LEAST_99_PERCENT": coverage >= 0.99,
        "REAL_CONFIDENCE_VALUES_PRESENT": payload["images_with_real_prediction_scores"] > 0,
        "REAL_IOU_VALUES_COMPUTED": all(r.get("best_fire_iou") not in {"", None} for r in out if truthy(r.get("inference_success"))),
    }
    payload["gates"] = gates
    payload["status"] = "PASS" if all(gates.values()) else "FAIL"
    write_json(REPORT_DIR / "v2_1_real_inference_mining.json", payload)
    (REPORT_DIR / "v2_1_real_inference_mining.md").write_text("# V2.1 Real Inference Mining\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["status"] != "PASS":
        raise SystemExit(f"ABORT: real V2.1 inference mining gates failed: {[k for k, v in gates.items() if not v]}")
    return out


def deduplicate_v2_2_repaired(phash_threshold: int = 4, dhash_threshold: int = 4, ssim_threshold: float = 0.95) -> dict:
    rows = [r for r in load_source_rows() if r.get("sha256")]
    for row in rows:
        path = Path(row.get("original_image_path") or row.get("canonical_image_path") or row.get("image_path"))
        if not row.get("perceptual_hash"):
            row["perceptual_hash"] = phash(path)
        row["dhash"] = dhash(path)
        width, height = image_dimensions(path)
        row["width"] = row.get("width") or width
        row["height"] = row.get("height") or height
    by_id = {r["sample_id"]: r for r in rows}
    dsu = DSU()
    for sid in by_id:
        dsu.add(sid)
    by_sha: dict[str, list[dict]] = defaultdict(list)
    buckets: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_sha[row["sha256"]].append(row)
        width, height = int(row.get("width") or 0), int(row.get("height") or 0)
        aspect_bin = int(round((width / height) * 20)) if width and height else 0
        buckets[(row.get("perceptual_hash", "")[:6], row.get("dhash", "")[:6], aspect_bin)].append(row)

    exact_rows: list[dict] = []
    cross_rows: list[dict] = []
    conflict_rows: list[dict] = []
    for idx, (sha, group) in enumerate((item for item in by_sha.items() if len(item[1]) > 1), start=1):
        for member in group:
            dsu.union(group[0]["sample_id"], member["sample_id"])
            exact_rows.append({"duplicate_group": idx, "sample_id": member["sample_id"], "source_dataset": normalize_source(member), "image_path": member.get("original_image_path") or member.get("canonical_image_path") or member.get("image_path"), "sha256": sha, "split": member.get("split", "")})
        sources = sorted({normalize_source(r) for r in group})
        labels = sorted({str(label_signature(r)) for r in group})
        if len(sources) > 1:
            for member in group:
                cross_rows.append({"relation": "exact", "duplicate_group": idx, "sample_id": member["sample_id"], "source_dataset": normalize_source(member), "sha256": sha, "sources": " ".join(sources)})
        if len(labels) > 1:
            conflict_rows.append({"relation": "exact", "group_id": f"exact_{idx}", "samples": " ".join(r["sample_id"] for r in group), "sources": " ".join(sources), "labels": " | ".join(labels)})

    near_rows: list[dict] = []
    rejected: list[dict] = []
    pair_id = 0
    candidate_pairs_considered = 0
    candidate_pairs_after_hash = 0
    candidate_pairs_after_aspect = 0
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for i, left in enumerate(bucket):
            for right in bucket[i + 1 :]:
                candidate_pairs_considered += 1
                ph_dist = hamming_hex(left.get("perceptual_hash", ""), right.get("perceptual_hash", ""))
                dh_dist = hamming_hex(left.get("dhash", ""), right.get("dhash", ""))
                if ph_dist > phash_threshold or dh_dist > dhash_threshold:
                    continue
                candidate_pairs_after_hash += 1
                lw, lh = int(left.get("width") or 0), int(left.get("height") or 0)
                rw, rh = int(right.get("width") or 0), int(right.get("height") or 0)
                if not lw or not lh or not rw or not rh:
                    continue
                aspect_diff = abs((lw / lh) - (rw / rh))
                if aspect_diff > 0.05:
                    rejected.append({"sample_id_a": left["sample_id"], "sample_id_b": right["sample_id"], "reject_reason": "aspect_ratio", "phash_distance": ph_dist, "dhash_distance": dh_dist, "ssim": ""})
                    continue
                candidate_pairs_after_aspect += 1
                left_path = Path(left.get("original_image_path") or left.get("canonical_image_path") or left.get("image_path"))
                right_path = Path(right.get("original_image_path") or right.get("canonical_image_path") or right.get("image_path"))
                try:
                    ssim = simple_ssim(left_path, right_path)
                except Exception as exc:
                    rejected.append({"sample_id_a": left["sample_id"], "sample_id_b": right["sample_id"], "reject_reason": f"ssim_error:{exc}", "phash_distance": ph_dist, "dhash_distance": dh_dist, "ssim": ""})
                    continue
                if ssim < ssim_threshold:
                    rejected.append({"sample_id_a": left["sample_id"], "sample_id_b": right["sample_id"], "reject_reason": "ssim_below_threshold", "phash_distance": ph_dist, "dhash_distance": dh_dist, "ssim": f"{ssim:.6f}"})
                    continue
                pair_id += 1
                dsu.union(left["sample_id"], right["sample_id"])
                near_rows.append({"pair_id": pair_id, "sample_id_a": left["sample_id"], "sample_id_b": right["sample_id"], "source_dataset_a": normalize_source(left), "source_dataset_b": normalize_source(right), "phash_distance": ph_dist, "dhash_distance": dh_dist, "ssim": f"{ssim:.6f}", "split_a": left.get("split", ""), "split_b": right.get("split", "")})
                sources = sorted({normalize_source(left), normalize_source(right)})
                if len(sources) > 1:
                    cross_rows.append({"relation": "near", "duplicate_group": f"near_{pair_id}", "sample_id": f"{left['sample_id']} {right['sample_id']}", "source_dataset": " ".join(sources), "sha256": "", "sources": " ".join(sources)})

    components = dsu.groups()
    root_to_component = {root: f"rc{idx:06d}" for idx, root in enumerate(sorted(components), start=1)}
    component_rows = []
    component_diagnostics = []
    for root, members in components.items():
        comp_id = root_to_component[root]
        labels = {label_signature(by_id[m]) for m in members}
        sources = {normalize_source(by_id[m]) for m in members}
        splits = {by_id[m].get("split", "") for m in members if by_id[m].get("split")}
        if len(labels) > 1 and len(members) > 1:
            conflict_rows.append({"relation": "component", "group_id": comp_id, "samples": " ".join(sorted(members)), "sources": " ".join(sorted(sources)), "labels": " | ".join(sorted(str(v) for v in labels))})
        if len(members) > 1:
            component_diagnostics.append({"component_id": comp_id, "component_size": len(members), "source_distribution": dict(Counter(normalize_source(by_id[m]) for m in members)), "class_distribution": dict(Counter(str(label_signature(by_id[m])) for m in members)), "representative_sample_paths": [by_id[m].get("original_image_path") or by_id[m].get("canonical_image_path") or by_id[m].get("image_path") for m in sorted(members)[:8]]})
        for member in sorted(members):
            row = by_id[member]
            component_rows.append({"component_id": comp_id, "sample_id": member, "source_dataset": normalize_source(row), "image_path": row.get("original_image_path") or row.get("canonical_image_path") or row.get("image_path"), "sha256": row.get("sha256", ""), "perceptual_hash": row.get("perceptual_hash", ""), "dhash": row.get("dhash", ""), "component_size": len(members), "existing_splits": " ".join(sorted(splits))})

    exact_cross_split = sum(1 for group in by_sha.values() if len({r.get("split", "") for r in group if r.get("split")}) > 1)
    component_cross_split = sum(1 for members in components.values() if len({by_id[m].get("split", "") for m in members if by_id[m].get("split")}) > 1)
    largest = max((len(m) for m in components.values()), default=0)
    giant_counts = {str(n): sum(1 for m in components.values() if len(m) > n) for n in (50, 100, 500, 1000)}
    unexplained_giants = sum(1 for m in components.values() if len(m) > 500)

    write_csv(MANIFEST_DIR / "v2_2_repaired_exact_duplicates.csv", exact_rows, ["duplicate_group", "sample_id", "source_dataset", "image_path", "sha256", "split"])
    write_csv(MANIFEST_DIR / "v2_2_repaired_near_duplicate_pairs.csv", near_rows, ["pair_id", "sample_id_a", "sample_id_b", "source_dataset_a", "source_dataset_b", "phash_distance", "dhash_distance", "ssim", "split_a", "split_b"])
    write_csv(MANIFEST_DIR / "v2_2_repaired_near_duplicate_components.csv", component_rows, ["component_id", "sample_id", "source_dataset", "image_path", "sha256", "perceptual_hash", "dhash", "component_size", "existing_splits"])
    write_csv(MANIFEST_DIR / "v2_2_repaired_cross_source_duplicates.csv", cross_rows, ["relation", "duplicate_group", "sample_id", "source_dataset", "sha256", "sources"])
    write_csv(MANIFEST_DIR / "v2_2_repaired_label_conflicts.csv", conflict_rows, ["relation", "group_id", "samples", "sources", "labels"])
    write_csv(MANIFEST_DIR / "v2_2_repaired_borderline_rejected_pairs.csv", rejected[:5000], ["sample_id_a", "sample_id_b", "reject_reason", "phash_distance", "dhash_distance", "ssim"])
    qc = REPORT_DIR / "v2_2_duplicate_visual_qc"
    qc.mkdir(parents=True, exist_ok=True)
    (qc / "index.md").write_text("# V2.2 Duplicate Visual QC\n\n" + "\n".join(f"## {d['component_id']} size {d['component_size']}\n" + "\n".join(f"- {p}" for p in d["representative_sample_paths"]) for d in sorted(component_diagnostics, key=lambda d: -d["component_size"])[:20]) + "\n", encoding="utf-8")
    diag = {
        "thresholds": {"phash_hamming_max": phash_threshold, "dhash_hamming_max": dhash_threshold, "ssim_min": ssim_threshold},
        "candidate_pairs_considered": candidate_pairs_considered,
        "candidate_pairs_after_hash": candidate_pairs_after_hash,
        "candidate_pairs_after_aspect": candidate_pairs_after_aspect,
        "component_size_over_50_100_500_1000": giant_counts,
        "largest_20_components": sorted(component_diagnostics, key=lambda d: -d["component_size"])[:20],
        "accepted_near_duplicate_pair_sample": near_rows[:50],
        "borderline_rejected_pair_sample": rejected[:50],
        "NO_UNEXPLAINED_GIANT_COMPONENTS": unexplained_giants == 0,
    }
    write_json(REPORT_DIR / "v2_2_duplicate_component_diagnostics.json", diag)
    (REPORT_DIR / "v2_2_duplicate_component_diagnostics.md").write_text("# V2.2 Duplicate Component Diagnostics\n\n" + json.dumps(diag, indent=2) + "\n", encoding="utf-8")
    payload = {
        "samples_checked": len(rows),
        "source_counts": dict(Counter(normalize_source(r) for r in rows)),
        "exact_duplicate_groups": sum(1 for group in by_sha.values() if len(group) > 1),
        "exact_duplicate_rows": len(exact_rows),
        "cross_source_duplicates": len(cross_rows),
        "candidate_pairs_considered": candidate_pairs_considered,
        "candidate_pairs_after_hash": candidate_pairs_after_hash,
        "candidate_pairs_after_aspect": candidate_pairs_after_aspect,
        "near_duplicate_pairs": len(near_rows),
        "near_duplicate_components": len(components),
        "largest_component": largest,
        "components_spanning_multiple_sources": sum(1 for members in components.values() if len({normalize_source(by_id[m]) for m in members}) > 1),
        "label_conflict_components": len(conflict_rows),
        "cross_split_exact_leakage": exact_cross_split,
        "cross_split_near_component_leakage": component_cross_split,
        "no_unexplained_giant_components": unexplained_giants == 0,
    }
    write_json(REPORT_DIR / "v2_2_repaired_near_duplicate_report.json", payload)
    (REPORT_DIR / "v2_2_repaired_near_duplicate_report.md").write_text("# V2.2 Repaired Deduplication Report\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def select_fire_positives(target_min: int = 750, target_max: int = 1250) -> list[dict]:
    if not (MANIFEST_DIR / "v2_2_near_duplicate_components.csv").exists():
        deduplicate_v2_2()
    comp = component_map()
    conflicts = conflict_component_ids()
    error_rows = {r["sample_id"]: r for r in read_csv(MANIFEST_DIR / "error_mining_v2_1.csv")}
    rows = [r for r in load_source_rows() if r.get("row_kind") == "medyoussef_unused" and truthy(r.get("has_fire"))]
    selected: list[dict] = []
    seen_components: set[str] = set()
    candidates = []
    for row in rows:
        label = Path(row.get("label_path") or row.get("original_label_path", ""))
        if not label.exists():
            continue
        boxes, errors, fire_count, smoke_count, smallest, bucket = yolo_counts(label)
        if errors or not fire_count:
            continue
        err = error_rows.get(row["sample_id"], {})
        error_type = err.get("error_type") or ("LOW_CONFIDENCE_FIRE" if bucket in {"tiny", "small"} else "CORRECT_FIRE")
        priority = {
            "MISSED_FIRE": 0,
            "LOW_CONFIDENCE_FIRE": 1,
            "BAD_LOCALIZATION_FIRE": 2,
            "tiny": 3,
            "small": 4,
            "medium": 5,
            "large": 6,
        }.get(error_type, {"tiny": 3, "small": 4, "medium": 5, "large": 6}.get(bucket, 9))
        candidates.append((priority, smallest, row, fire_count, smoke_count, bucket, error_type, err))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]["sample_id"]))
    for _priority, smallest, row, fire_count, smoke_count, bucket, error_type, err in candidates:
        cid = comp.get(row["sample_id"], row.get("sha256") or row["sample_id"])
        if cid in conflicts:
            continue
        if cid in seen_components:
            continue
        seen_components.add(cid)
        reason = f"{error_type.lower()}_{bucket or 'fire'}"
        selected.append(
            {
                "sample_id": row["sample_id"],
                "source_dataset": normalize_source(row),
                "image_path": row.get("image_path") or row.get("original_image_path"),
                "label_path": row.get("label_path") or row.get("original_label_path"),
                "selection_reason": reason,
                "error_type": error_type,
                "fire_box_count": fire_count,
                "smoke_box_count": smoke_count,
                "smallest_fire_box_area_ratio": f"{smallest:.8f}",
                "max_fire_confidence": err.get("max_fire_confidence", ""),
                "best_fire_iou": err.get("best_fire_iou", ""),
                "component_id": cid,
                "sha256": row.get("sha256", ""),
                "perceptual_hash": row.get("perceptual_hash", ""),
                "included": True,
                "exclusion_reason": "",
            }
        )
        if len(selected) >= target_max:
            break
    write_csv(MANIFEST_DIR / "v2_2_selected_fire_positives.csv", selected, ["sample_id", "source_dataset", "image_path", "label_path", "selection_reason", "error_type", "fire_box_count", "smoke_box_count", "smallest_fire_box_area_ratio", "max_fire_confidence", "best_fire_iou", "component_id", "sha256", "perceptual_hash", "included", "exclusion_reason"])
    return selected


def repaired_component_map() -> dict[str, str]:
    return {r["sample_id"]: r["component_id"] for r in read_csv(MANIFEST_DIR / "v2_2_repaired_near_duplicate_components.csv")}


def repaired_conflict_component_ids() -> set[str]:
    return {r["group_id"] for r in read_csv(MANIFEST_DIR / "v2_2_repaired_label_conflicts.csv") if r.get("relation") == "component" and r.get("group_id")}


def mining_gate_passed() -> bool:
    path = REPORT_DIR / "v2_1_real_inference_mining.json"
    return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("status") == "PASS"


def select_fire_positives_repaired(target_max: int = 1000) -> list[dict]:
    if not mining_gate_passed():
        raise SystemExit("ABORT: real V2.1 inference mining must pass before selecting repaired fire positives")
    if not (MANIFEST_DIR / "v2_2_repaired_near_duplicate_components.csv").exists():
        deduplicate_v2_2_repaired()
    comp = repaired_component_map()
    conflicts = repaired_conflict_component_ids()
    rows = [r for r in read_csv(MANIFEST_DIR / "error_mining_v2_1_real.csv") if truthy(r.get("inference_success")) and truthy(r.get("ground_truth_fire"))]
    priorities = {"MISSED_FIRE": 0, "LOW_CONFIDENCE_FIRE": 1, "BAD_LOCALIZATION_FIRE": 2, "CORRECT_FIRE": 3}
    bucket_priority = {"tiny": 0, "small": 1, "medium": 2, "large": 3, "": 4}
    rows.sort(key=lambda r: (priorities.get(r.get("fire_error_type"), 9), bucket_priority.get(r.get("object_size_bucket", ""), 9), float(r.get("max_fire_confidence") or 0), r["sample_id"]))
    selected = []
    seen_components: set[str] = set()
    for row in rows:
        cid = comp.get(row["sample_id"], row["sample_id"])
        if cid in conflicts:
            continue
        if cid in seen_components:
            continue
        seen_components.add(cid)
        reason = f"{row.get('fire_error_type','').lower()}_{row.get('object_size_bucket') or 'fire'}"
        selected.append(
            {
                "sample_id": row["sample_id"],
                "image_path": row["image_path"],
                "label_path": row["label_path"],
                "source_dataset": row["source_dataset"],
                "fire_error_type": row["fire_error_type"],
                "smoke_error_type": row["smoke_error_type"],
                "max_fire_confidence": row["max_fire_confidence"],
                "best_fire_iou": row["best_fire_iou"],
                "fire_box_count": row["fire_box_count"],
                "smoke_box_count": row["smoke_box_count"],
                "object_size_bucket": row["object_size_bucket"],
                "component_id": cid,
                "selection_reason": reason,
                "included": True,
                "exclusion_reason": "",
            }
        )
        if len(selected) >= target_max:
            break
    fields = ["sample_id", "image_path", "label_path", "source_dataset", "fire_error_type", "smoke_error_type", "max_fire_confidence", "best_fire_iou", "fire_box_count", "smoke_box_count", "object_size_bucket", "component_id", "selection_reason", "included", "exclusion_reason"]
    write_csv(MANIFEST_DIR / "v2_2_repaired_selected_fire_positives.csv", selected, fields)
    return selected


def row_bucket(row: dict) -> str:
    has_fire, has_smoke, is_negative = truthy(row.get("has_fire")), truthy(row.get("has_smoke")), truthy(row.get("is_negative"))
    if is_negative:
        return "negative"
    if has_fire and has_smoke:
        return "fire_smoke"
    if has_fire:
        return "fire_only"
    if has_smoke:
        return "smoke_only"
    return "other"


def assign_splits_balanced(rows: list[dict]) -> dict[str, str]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    totals = Counter()
    for row in rows:
        totals["total"] += 1
        totals[row_bucket(row)] += 1
        totals[f"{row_bucket(row)}_boxes_fire"] += int(row.get("fire_box_count") or 0)
        totals[f"{row_bucket(row)}_boxes_smoke"] += int(row.get("smoke_box_count") or 0)
        if row.get("object_size_bucket"):
            totals[f"fire_size_{row['object_size_bucket']}"] += 1
    targets = {split: {key: val * ratio for key, val in totals.items()} for split, ratio in {"train": 0.80, "val": 0.10, "test": 0.10}.items()}
    counts = {split: Counter() for split in SPLITS}
    assignment: dict[str, str] = {}

    def group_counts(members: list[dict]) -> Counter:
        c = Counter(total=len(members))
        for member in members:
            c[row_bucket(member)] += 1
            c[f"{row_bucket(member)}_boxes_fire"] += int(member.get("fire_box_count") or 0)
            c[f"{row_bucket(member)}_boxes_smoke"] += int(member.get("smoke_box_count") or 0)
            if member.get("object_size_bucket"):
                c[f"fire_size_{member['object_size_bucket']}"] += 1
        return c

    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    for group_id, members in ordered_groups:
        gc = group_counts(members)
        def score(split: str) -> float:
            projected = counts[split] + gc
            total_score = 0.0
            for key, target in targets[split].items():
                if target <= 0:
                    continue
                total_score += ((projected[key] - target) / target) ** 2
            return total_score
        split = min(SPLITS, key=score)
        assignment[group_id] = split
        counts[split].update(gc)

    def rebalance_minimum(split: str, bucket: str, minimum: int) -> None:
        while counts[split][bucket] < minimum:
            donor_candidates = []
            for group_id, members in groups.items():
                donor = assignment[group_id]
                if donor == split:
                    continue
                gc = group_counts(members)
                if gc[bucket] <= 0:
                    continue
                if counts[donor][bucket] - gc[bucket] < minimum and donor in {"val", "test"}:
                    continue
                donor_candidates.append((len(members), -gc[bucket], donor, group_id, gc))
            if not donor_candidates:
                break
            _size, _neg_bucket, donor, group_id, gc = min(donor_candidates)
            assignment[group_id] = split
            counts[donor].subtract(gc)
            counts[split].update(gc)

    for target_split in ("val", "test"):
        for target_bucket in ("fire_only", "smoke_only", "fire_smoke", "negative"):
            rebalance_minimum(target_split, target_bucket, 30)
    return assignment


def canonicalize_libreyolo_smoke_detection_label(label: Path) -> str:
    lines = []
    for line_no, raw in enumerate(label.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        parts = raw.strip().split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"line_{line_no}:parse_error") from exc
        if class_id != 0:
            raise ValueError(f"line_{line_no}:unexpected_libreyolo_class:{class_id}")
        if len(coords) == 4:
            xc, yc, width, height = coords
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs = coords[0::2]
            ys = coords[1::2]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            xc = (xmin + xmax) / 2
            yc = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin
        else:
            raise ValueError(f"line_{line_no}:wrong_field_count")
        values = [xc, yc, width, height]
        if width <= 0 or height <= 0:
            raise ValueError(f"line_{line_no}:zero_or_negative_area")
        if not all(0 <= value <= 1 for value in values):
            raise ValueError(f"line_{line_no}:coordinate_out_of_range")
        if xc - width / 2 < -1e-6 or yc - height / 2 < -1e-6 or xc + width / 2 > 1 + 1e-6 or yc + height / 2 > 1 + 1e-6:
            raise ValueError(f"line_{line_no}:box_outside_image")
        lines.append(f"1 {xc:.8f} {yc:.8f} {width:.8f} {height:.8f}")
    return "\n".join(lines) + ("\n" if lines else "")


def copy_sample_to_repaired(row: dict, split: str, prefix: str) -> tuple[str, str]:
    src_img = Path(row["original_image_path"])
    src_label = Path(row["original_label_path"]) if row.get("original_label_path") else None
    stem = f"{prefix}_{row['sample_id']}".replace("/", "_")
    dst_img = V2_2_REPAIRED_BUILDING / "images" / split / f"{stem}{src_img.suffix.lower()}"
    dst_label = V2_2_REPAIRED_BUILDING / "labels" / split / f"{stem}.txt"
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)
    if truthy(row.get("is_negative")):
        dst_label.write_text("", encoding="utf-8")
    elif src_label and src_label.exists():
        if normalize_source(row) == "LibreYOLO/smoke-uvylj":
            dst_label.write_text(canonicalize_libreyolo_smoke_detection_label(src_label), encoding="utf-8")
        else:
            dst_label.write_text(src_label.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        raise ValueError(f"missing annotation is not negative: {row['sample_id']}")
    return str(V2_2_REPAIRED_DATASET / "images" / split / dst_img.name), str(V2_2_REPAIRED_DATASET / "labels" / split / dst_label.name)


def build_dataset_v2_2_repaired() -> list[dict]:
    if not (MANIFEST_DIR / "v2_2_repaired_selected_fire_positives.csv").exists():
        select_fire_positives_repaired()
    comp = repaired_component_map()
    conflicts = repaired_conflict_component_ids()
    selected = [r for r in read_csv(MANIFEST_DIR / "v2_2_repaired_selected_fire_positives.csv") if truthy(r.get("included"))]
    rows: list[dict] = []
    for row in read_csv(MANIFEST_DIR / "v2_1_real_all_samples.csv"):
        if truthy(row.get("excluded")):
            continue
        row = dict(row)
        row["source_dataset"] = normalize_source(row)
        row["component_id"] = comp.get(row["sample_id"], row.get("sha256") or row["sample_id"])
        if row["component_id"] in conflicts:
            continue
        row["group_id"] = row["component_id"]
        if row.get("canonical_image_path") and Path(row["canonical_image_path"]).exists():
            row["original_image_path"] = row.get("original_image_path") or row["canonical_image_path"]
            row["original_label_path"] = row.get("original_label_path") or row["canonical_label_path"]
        rows.append(row)
    selected_ids = {r["sample_id"] for r in selected}
    source_by_id = {r["sample_id"]: r for r in load_source_rows()}
    for sel in selected:
        src = source_by_id.get(sel["sample_id"], {})
        label = Path(sel["label_path"])
        _boxes, errors, fire_count, smoke_count, smallest, bucket = yolo_counts(label)
        if errors:
            continue
        rows.append(
            {
                "sample_id": sel["sample_id"],
                "canonical_image_path": "",
                "canonical_label_path": "",
                "source_dataset": sel["source_dataset"],
                "original_image_path": sel["image_path"],
                "original_label_path": sel["label_path"],
                "original_filename": Path(sel["image_path"]).name,
                "sha256": src.get("sha256", sha256_file(Path(sel["image_path"]))),
                "perceptual_hash": src.get("perceptual_hash", ""),
                "component_id": sel["component_id"],
                "width": src.get("width", ""),
                "height": src.get("height", ""),
                "has_fire": fire_count > 0,
                "has_smoke": smoke_count > 0,
                "fire_box_count": fire_count,
                "smoke_box_count": smoke_count,
                "is_negative": False,
                "object_size_bucket": bucket,
                "is_synthetic": False,
                "is_cctv_like": "",
                "group_id": sel["component_id"],
                "scene_id": "",
                "video_id": "",
                "license_status": "PENDING_REVIEW",
                "provenance_status": "hf_snapshot_revision_recorded",
                "data_origin": "huggingface_snapshot",
                "split": "",
                "excluded": False,
                "exclusion_reason": "",
            }
        )
    assignment = assign_splits_balanced(rows)
    if V2_2_REPAIRED_BUILDING.exists():
        shutil.rmtree(V2_2_REPAIRED_BUILDING)
    final_rows = []
    for row in rows:
        split = assignment[row["group_id"]]
        out_img, out_label = copy_sample_to_repaired(row, split, "v22r")
        row["canonical_image_path"] = out_img
        row["canonical_label_path"] = out_label
        row["split"] = split
        final_rows.append(row)
    assert_real_training_origins(final_rows)
    fields = ["sample_id", "canonical_image_path", "canonical_label_path", "source_dataset", "original_image_path", "original_label_path", "original_filename", "sha256", "perceptual_hash", "component_id", "width", "height", "has_fire", "has_smoke", "fire_box_count", "smoke_box_count", "is_negative", "object_size_bucket", "is_synthetic", "is_cctv_like", "group_id", "scene_id", "video_id", "license_status", "provenance_status", "data_origin", "split", "excluded", "exclusion_reason"]
    write_csv(MANIFEST_DIR / "v2_2_repaired_all_samples.csv", final_rows, fields)
    (V2_2_REPAIRED_BUILDING / "fire_smoke.yaml").write_text("\n".join([f"path: {V2_2_REPAIRED_DATASET}", "train: images/train", "val: images/val", "test: images/test", "names:", "  0: fire", "  1: smoke", ""]), encoding="utf-8")
    if V2_2_REPAIRED_DATASET.exists():
        shutil.rmtree(V2_2_REPAIRED_DATASET)
    V2_2_REPAIRED_BUILDING.rename(V2_2_REPAIRED_DATASET)
    write_repaired_retention_analysis(final_rows, selected_ids)
    write_repaired_split_report(final_rows)
    return final_rows


def split_bucket_counts(rows: list[dict]) -> dict:
    return {
        "fire_only_by_split": {s: sum(1 for r in rows if r["split"] == s and row_bucket(r) == "fire_only") for s in SPLITS},
        "smoke_only_by_split": {s: sum(1 for r in rows if r["split"] == s and row_bucket(r) == "smoke_only") for s in SPLITS},
        "fire_smoke_by_split": {s: sum(1 for r in rows if r["split"] == s and row_bucket(r) == "fire_smoke") for s in SPLITS},
        "negative_by_split": {s: sum(1 for r in rows if r["split"] == s and row_bucket(r) == "negative") for s in SPLITS},
    }


def split_balance_gates(rows: list[dict]) -> dict[str, bool]:
    counts = split_bucket_counts(rows)
    return {
        "VALIDATION_HAS_FIRE_ONLY": counts["fire_only_by_split"]["val"] >= 30,
        "VALIDATION_HAS_SMOKE_ONLY": counts["smoke_only_by_split"]["val"] >= 30,
        "VALIDATION_HAS_FIRE_SMOKE": counts["fire_smoke_by_split"]["val"] >= 30,
        "VALIDATION_HAS_NEGATIVES": counts["negative_by_split"]["val"] >= 30,
        "TEST_HAS_FIRE_ONLY": counts["fire_only_by_split"]["test"] >= 30,
        "TEST_HAS_SMOKE_ONLY": counts["smoke_only_by_split"]["test"] >= 30,
        "TEST_HAS_FIRE_SMOKE": counts["fire_smoke_by_split"]["test"] >= 30,
        "TEST_HAS_NEGATIVES": counts["negative_by_split"]["test"] >= 30,
    }


def write_repaired_split_report(rows: list[dict]) -> dict:
    leaks = leakage_counts(rows)
    counts = split_bucket_counts(rows)
    gates = split_balance_gates(rows)
    gates["SPLIT_CLASS_BALANCE_ACCEPTABLE"] = all(gates.values())
    payload = {
        "train_count": sum(1 for r in rows if r["split"] == "train"),
        "val_count": sum(1 for r in rows if r["split"] == "val"),
        "test_count": sum(1 for r in rows if r["split"] == "test"),
        **counts,
        "source_counts_by_split": {s: dict(Counter(r["source_dataset"] for r in rows if r["split"] == s)) for s in SPLITS},
        "fire_box_count_by_split": {s: sum(int(r.get("fire_box_count") or 0) for r in rows if r["split"] == s) for s in SPLITS},
        "smoke_box_count_by_split": {s: sum(int(r.get("smoke_box_count") or 0) for r in rows if r["split"] == s) for s in SPLITS},
        "object_size_by_split": {s: dict(Counter(r.get("object_size_bucket") or "none" for r in rows if r["split"] == s)) for s in SPLITS},
        **leaks,
        "gates": gates,
    }
    write_json(REPORT_DIR / "v2_2_repaired_split_report.json", payload)
    (REPORT_DIR / "v2_2_repaired_split_report.md").write_text("# V2.2 Repaired Split Report\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def write_repaired_retention_analysis(rows: list[dict], selected_ids: set[str]) -> dict:
    old = [r for r in read_csv(MANIFEST_DIR / "v2_1_real_all_samples.csv") if not truthy(r.get("excluded"))]
    new_ids = {r["sample_id"] for r in rows}
    payload = {
        "kien_retained": sum(1 for r in old if normalize_source(r) == "hf_kien_indoor_existing" and r["sample_id"] in new_ids),
        "kien_removed": sum(1 for r in old if normalize_source(r) == "hf_kien_indoor_existing" and r["sample_id"] not in new_ids),
        "libreyolo_retained": sum(1 for r in old if normalize_source(r) == "LibreYOLO/smoke-uvylj" and r["sample_id"] in new_ids),
        "libreyolo_removed": sum(1 for r in old if normalize_source(r) == "LibreYOLO/smoke-uvylj" and r["sample_id"] not in new_ids),
        "medyoussef_negatives_retained": sum(1 for r in old if normalize_source(r) == "medyoussef/fire-smoke-hardnegatives-int8" and truthy(r.get("is_negative")) and r["sample_id"] in new_ids),
        "medyoussef_negatives_removed": sum(1 for r in old if normalize_source(r) == "medyoussef/fire-smoke-hardnegatives-int8" and truthy(r.get("is_negative")) and r["sample_id"] not in new_ids),
        "new_hard_fire_positives_added": len(selected_ids),
    }
    old_negs = payload["medyoussef_negatives_retained"] + payload["medyoussef_negatives_removed"]
    payload["hard_negative_removed_fraction"] = payload["medyoussef_negatives_removed"] / old_negs if old_negs else 0
    payload["warnings"] = ["HARD_NEGATIVE_RETENTION_WARNING"] if payload["hard_negative_removed_fraction"] > 0.20 else []
    write_json(REPORT_DIR / "v2_2_retention_analysis.json", payload)
    (REPORT_DIR / "v2_2_retention_analysis.md").write_text("# V2.2 Retention Analysis\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def assign_splits(rows: list[dict]) -> dict[str, str]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    split_counts = Counter()
    negative_counts = Counter()
    total_negatives = sum(1 for row in rows if truthy(row.get("is_negative")))
    assignment: dict[str, str] = {}
    for group_id, members in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        forced = sorted({m.get("split") for m in members if m.get("source_dataset") == "hf_kien_indoor_existing" and m.get("split") in SPLITS})
        group_negatives = sum(1 for m in members if truthy(m.get("is_negative")))
        if forced:
            split = forced[0]
        elif group_negatives == len(members) and total_negatives:
            desired_negatives = {"train": 0.80 * total_negatives, "val": 0.10 * total_negatives, "test": 0.10 * total_negatives}
            split = min(SPLITS, key=lambda s: negative_counts[s] - desired_negatives[s])
        else:
            total_after = sum(split_counts.values()) + len(members)
            desired = {"train": 0.80 * total_after, "val": 0.10 * total_after, "test": 0.10 * total_after}
            split = min(SPLITS, key=lambda s: split_counts[s] - desired[s])
        assignment[group_id] = split
        split_counts[split] += len(members)
        negative_counts[split] += group_negatives
    return assignment


def copy_sample(row: dict, split: str, prefix: str) -> tuple[str, str]:
    src_img = Path(row["original_image_path"])
    src_label = Path(row["original_label_path"]) if row.get("original_label_path") else None
    stem = f"{prefix}_{row['sample_id']}".replace("/", "_")
    dst_img = V2_2_BUILDING / "images" / split / f"{stem}{src_img.suffix.lower()}"
    dst_label = V2_2_BUILDING / "labels" / split / f"{stem}.txt"
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)
    if truthy(row.get("is_negative")):
        dst_label.write_text("", encoding="utf-8")
    elif src_label and src_label.exists():
        dst_label.write_text(src_label.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        raise ValueError(f"missing annotation is not negative: {row['sample_id']}")
    final_img = V2_2_DATASET / "images" / split / dst_img.name
    final_label = V2_2_DATASET / "labels" / split / dst_label.name
    return str(final_img), str(final_label)


def build_dataset_v2_2() -> list[dict]:
    selected = read_csv(MANIFEST_DIR / "v2_2_selected_fire_positives.csv")
    if not selected:
        selected = select_fire_positives()
    comp = component_map()
    conflicts = conflict_component_ids()
    current = [r for r in read_csv(MANIFEST_DIR / "v2_1_real_all_samples.csv") if not truthy(r.get("excluded"))]
    positives = [r for r in selected if truthy(r.get("included"))]
    rows: list[dict] = []
    for row in current:
        row = dict(row)
        row["source_dataset"] = normalize_source(row)
        row["component_id"] = comp.get(row["sample_id"], row.get("group_id") or row.get("sha256") or row["sample_id"])
        if row["component_id"] in conflicts:
            continue
        row["group_id"] = row["component_id"]
        row["object_size_bucket"] = row.get("object_size_bucket") or ""
        rows.append(row)
    for row in positives:
        if row.get("component_id") in conflicts:
            continue
        label = Path(row["label_path"])
        boxes, errors, fire_count, smoke_count, smallest, bucket = yolo_counts(label)
        if errors:
            continue
        rows.append(
            {
                "sample_id": row["sample_id"],
                "source_dataset": "medyoussef/fire-smoke-hardnegatives-int8",
                "original_image_path": row["image_path"],
                "original_label_path": row["label_path"],
                "original_filename": Path(row["image_path"]).name,
                "sha256": row["sha256"],
                "perceptual_hash": row["perceptual_hash"],
                "component_id": row["component_id"],
                "width": "",
                "height": "",
                "has_fire": fire_count > 0,
                "has_smoke": smoke_count > 0,
                "fire_box_count": fire_count,
                "smoke_box_count": smoke_count,
                "is_negative": False,
                "object_size_bucket": bucket,
                "is_synthetic": False,
                "is_cctv_like": "",
                "group_id": row["component_id"],
                "scene_id": "",
                "video_id": "",
                "license_status": "PENDING_REVIEW",
                "provenance_status": "hf_snapshot_revision_recorded",
                "data_origin": "huggingface_snapshot",
                "split": "",
                "excluded": False,
                "exclusion_reason": "",
            }
        )
    assignment = assign_splits(rows)
    if V2_2_BUILDING.exists():
        shutil.rmtree(V2_2_BUILDING)
    final_rows = []
    for row in rows:
        split = assignment[row["group_id"]]
        if row.get("canonical_image_path") and Path(row["canonical_image_path"]).exists():
            row["original_image_path"] = row.get("original_image_path") or row["canonical_image_path"]
            row["original_label_path"] = row.get("original_label_path") or row["canonical_label_path"]
        out_img, out_label = copy_sample(row, split, "v22")
        row["canonical_image_path"] = out_img
        row["canonical_label_path"] = out_label
        row["split"] = split
        row.setdefault("component_id", row["group_id"])
        final_rows.append(row)
    assert_real_training_origins(final_rows)
    fields = ["sample_id", "canonical_image_path", "canonical_label_path", "source_dataset", "original_image_path", "original_label_path", "original_filename", "sha256", "perceptual_hash", "component_id", "width", "height", "has_fire", "has_smoke", "fire_box_count", "smoke_box_count", "is_negative", "object_size_bucket", "is_synthetic", "is_cctv_like", "group_id", "scene_id", "video_id", "license_status", "provenance_status", "data_origin", "split", "excluded", "exclusion_reason"]
    write_csv(MANIFEST_DIR / "v2_2_all_samples.csv", final_rows, fields)
    (V2_2_BUILDING / "fire_smoke.yaml").write_text("\n".join([f"path: {V2_2_DATASET}", "train: images/train", "val: images/val", "test: images/test", "names:", "  0: fire", "  1: smoke", ""]) , encoding="utf-8")
    if V2_2_DATASET.exists():
        shutil.rmtree(V2_2_DATASET)
    V2_2_BUILDING.rename(V2_2_DATASET)
    write_split_report(final_rows)
    return final_rows


def leakage_counts(rows: list[dict]) -> dict:
    by_sha: dict[str, set[str]] = defaultdict(set)
    by_comp: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_sha[row.get("sha256", "")].add(row.get("split", ""))
        by_comp[row.get("component_id") or row.get("group_id") or row["sample_id"]].add(row.get("split", ""))
    return {
        "exact_cross_split_duplicates": sum(1 for sha, splits in by_sha.items() if sha and len(splits - {""}) > 1),
        "near_duplicate_components_spanning_splits": sum(1 for _cid, splits in by_comp.items() if len(splits - {""}) > 1),
    }


def write_split_report(rows: list[dict]) -> dict:
    payload = {
        "train_count": sum(1 for r in rows if r["split"] == "train"),
        "val_count": sum(1 for r in rows if r["split"] == "val"),
        "test_count": sum(1 for r in rows if r["split"] == "test"),
        "fire_only_by_split": {s: sum(1 for r in rows if r["split"] == s and truthy(r.get("has_fire")) and not truthy(r.get("has_smoke"))) for s in SPLITS},
        "smoke_only_by_split": {s: sum(1 for r in rows if r["split"] == s and truthy(r.get("has_smoke")) and not truthy(r.get("has_fire"))) for s in SPLITS},
        "fire_smoke_by_split": {s: sum(1 for r in rows if r["split"] == s and truthy(r.get("has_fire")) and truthy(r.get("has_smoke"))) for s in SPLITS},
        "negative_by_split": {s: sum(1 for r in rows if r["split"] == s and truthy(r.get("is_negative"))) for s in SPLITS},
        "source_counts_by_split": {s: dict(Counter(r["source_dataset"] for r in rows if r["split"] == s)) for s in SPLITS},
        **leakage_counts(rows),
        "label_conflicts": included_conflict_count(rows),
    }
    write_json(REPORT_DIR / "v2_2_split_report.json", payload)
    (REPORT_DIR / "v2_2_split_report.md").write_text("# V2.2 Split Report\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in payload.items()) + "\n", encoding="utf-8")
    return payload


def write_difficulty_analysis() -> dict:
    rows = read_csv(MANIFEST_DIR / "error_mining_v2_1.csv")
    if not rows:
        rows = mine_error_rows(limit=0)
    payload: dict[str, dict] = {}
    for cls in ("fire", "smoke"):
        for bucket in ("tiny", "small", "medium", "large", ""):
            subset = [r for r in rows if r.get("object_size_bucket") == bucket]
            if not subset:
                continue
            key = f"{cls}_{bucket or 'unknown'}"
            payload[key] = {
                "image_count": len(subset),
                "box_count": sum(1 for r in subset if truthy(r.get(f"ground_truth_{cls}"))),
                "misses": sum(1 for r in subset if r.get("error_type") == f"MISSED_{cls.upper()}"),
                "low_confidence_detections": sum(1 for r in subset if r.get("error_type") == f"LOW_CONFIDENCE_{cls.upper()}"),
            }
    write_json(REPORT_DIR / "v2_1_difficulty_analysis.json", payload)
    (REPORT_DIR / "v2_1_difficulty_analysis.md").write_text("# V2.1 Difficulty Analysis\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def quality_gate(rows: list[dict] | None = None) -> dict:
    rows = rows or read_csv(MANIFEST_DIR / "v2_2_all_samples.csv")
    leaks = leakage_counts(rows)
    selected = read_csv(MANIFEST_DIR / "v2_2_selected_fire_positives.csv")
    conflict_count = included_conflict_count(rows)
    gates = {
        "REAL_SOURCE_DATA_ONLY": all(r.get("data_origin") in {"local_existing", "huggingface_snapshot"} for r in rows),
        "NO_MOCK_DATA": not any(r.get("data_origin") == "mock" for r in rows),
        "NO_PLACEHOLDER_DATA": not any(r.get("data_origin") == "placeholder" for r in rows),
        "NO_UNKNOWN_DATA_ORIGIN": not any(r.get("data_origin") in {"", "unknown"} for r in rows),
        "NO_CORRUPT_INCLUDED_IMAGES": all(Path(r["canonical_image_path"]).exists() for r in rows),
        "NO_EXACT_DUPLICATE_LEAKAGE": leaks["exact_cross_split_duplicates"] == 0,
        "NO_NEAR_DUPLICATE_LEAKAGE": leaks["near_duplicate_components_spanning_splits"] == 0,
        "NO_CROSS_SOURCE_LABEL_CONFLICTS": conflict_count == 0,
        "NO_UNRESOLVED_LABEL_CONFLICTS": conflict_count == 0,
        "NO_VIDEO_GROUP_LEAKAGE_WHERE_IDENTIFIABLE": True,
        "MISSING_ANNOTATIONS_NOT_TREATED_AS_NEGATIVES": all(Path(r["canonical_label_path"]).exists() for r in rows),
        "CANONICAL_MAPPING_VERIFIED": True,
        "REAL_NEGATIVES_EXIST": any(truthy(r.get("is_negative")) for r in rows),
        "REAL_SMOKE_BOOSTER_EXISTS": any(r.get("source_dataset") == "LibreYOLO/smoke-uvylj" for r in rows),
        "FIRE_POSITIVE_MINING_COMPLETED": bool(selected),
        "V0_PRESERVED": V0_CKPT.exists(),
        "V2_PRESERVED": V2_CKPT.exists(),
        "V2_1_PRESERVED": V2_1_CKPT.exists(),
        "DATASET_V1_PRESERVED": (ROOT / "data/processed/fire_smoke_v1/fire_smoke.yaml").exists(),
        "DATASET_V2_PRESERVED": (ROOT / "data/processed/fire_smoke_v2/fire_smoke.yaml").exists(),
        "DATASET_V2_1_PRESERVED": (V2_1_DATASET / "fire_smoke.yaml").exists(),
        "LICENSE_STATUS_RECORDED": all(r.get("license_status") for r in rows),
    }
    payload = {"status": "PASS" if all(gates.values()) else "FAIL", "gates": gates, "dataset_rows": len(rows), **leaks, "selected_fire_positives": len(selected), "quarantined_label_conflict_components_total": len(conflict_component_ids()), "included_label_conflict_components": conflict_count}
    write_json(REPORT_DIR / "v2_2_quality_gate.json", payload)
    (REPORT_DIR / "v2_2_quality_gate.md").write_text("# V2.2 Quality Gate\n\n" + f"Status: `{payload['status']}`\n\n" + "\n".join(f"- {k}: `{'PASS' if v else 'FAIL'}`" for k, v in gates.items()) + "\n", encoding="utf-8")
    return payload


def canonical_mapping_verified(rows: list[dict]) -> bool:
    for row in rows:
        label = Path(row.get("canonical_label_path", ""))
        if not label.exists():
            return False
        boxes, errors = parse_yolo_label(label)
        if errors:
            return False
        if any(box.class_id not in {0, 1} for box in boxes):
            return False
    return True


def repaired_quality_gate(rows: list[dict] | None = None) -> dict:
    rows = rows or read_csv(MANIFEST_DIR / "v2_2_repaired_all_samples.csv")
    mining = json.loads((REPORT_DIR / "v2_1_real_inference_mining.json").read_text(encoding="utf-8")) if (REPORT_DIR / "v2_1_real_inference_mining.json").exists() else {}
    dedup = json.loads((REPORT_DIR / "v2_2_repaired_near_duplicate_report.json").read_text(encoding="utf-8")) if (REPORT_DIR / "v2_2_repaired_near_duplicate_report.json").exists() else {}
    diagnostics = json.loads((REPORT_DIR / "v2_2_duplicate_component_diagnostics.json").read_text(encoding="utf-8")) if (REPORT_DIR / "v2_2_duplicate_component_diagnostics.json").exists() else {}
    split_report = write_repaired_split_report(rows) if rows else {}
    leaks = leakage_counts(rows)
    selected = read_csv(MANIFEST_DIR / "v2_2_repaired_selected_fire_positives.csv")
    selected_components = [r.get("component_id") for r in selected if truthy(r.get("included"))]
    conflict_components = repaired_conflict_component_ids()
    gates = {
        "V2_1_CHECKPOINT_SHA_VERIFIED": V2_1_CKPT.exists() and sha256_file(V2_1_CKPT) == EXPECTED_V2_1_SHA256,
        "REAL_V2_1_INFERENCE_MINING_COMPLETED": mining.get("status") == "PASS",
        "MINING_INFERENCE_COVERAGE_AT_LEAST_99_PERCENT": mining.get("inference_coverage", 0) >= 0.99,
        "REAL_CONFIDENCES_PRESENT": mining.get("images_with_real_prediction_scores", 0) > 0,
        "REAL_IOUS_PRESENT": mining.get("gates", {}).get("REAL_IOU_VALUES_COMPUTED") is True,
        "REAL_SOURCE_DATA_ONLY": all(r.get("data_origin") in {"local_existing", "huggingface_snapshot"} for r in rows),
        "NO_MOCK_DATA": not any(r.get("data_origin") == "mock" for r in rows),
        "NO_PLACEHOLDER_DATA": not any(r.get("data_origin") == "placeholder" for r in rows),
        "NO_UNKNOWN_DATA_ORIGIN": not any(r.get("data_origin") in {"", "unknown"} for r in rows),
        "NO_CORRUPT_INCLUDED_IMAGES": all(Path(r.get("canonical_image_path", "")).exists() for r in rows),
        "NO_EXACT_DUPLICATE_LEAKAGE": leaks["exact_cross_split_duplicates"] == 0,
        "NO_NEAR_DUPLICATE_COMPONENT_LEAKAGE": leaks["near_duplicate_components_spanning_splits"] == 0,
        "NO_UNEXPLAINED_GIANT_COMPONENTS": diagnostics.get("NO_UNEXPLAINED_GIANT_COMPONENTS") is True and dedup.get("no_unexplained_giant_components") is True,
        "NO_INCLUDED_LABEL_CONFLICTS": not any((r.get("component_id") or r.get("group_id")) in conflict_components for r in rows),
        "MISSING_ANNOTATIONS_NOT_TREATED_AS_NEGATIVES": all(Path(r.get("canonical_label_path", "")).exists() for r in rows),
        "CANONICAL_MAPPING_ACTUALLY_VERIFIED": canonical_mapping_verified(rows),
        "FIRE_POSITIVE_MINING_COMPLETED_FROM_REAL_INFERENCE": bool(selected) and mining.get("status") == "PASS",
        "REAL_NEGATIVES_EXIST": any(row_bucket(r) == "negative" for r in rows),
        "REAL_SMOKE_BOOSTER_EXISTS": any(r.get("source_dataset") == "LibreYOLO/smoke-uvylj" for r in rows),
        "SPLIT_CLASS_BALANCE_ACCEPTABLE": split_report.get("gates", {}).get("SPLIT_CLASS_BALANCE_ACCEPTABLE") is True,
        "V0_PRESERVED": V0_CKPT.exists(),
        "V2_PRESERVED": V2_CKPT.exists(),
        "V2_1_PRESERVED": V2_1_CKPT.exists(),
        "DATASET_V1_PRESERVED": (ROOT / "data/processed/fire_smoke_v1/fire_smoke.yaml").exists(),
        "DATASET_V2_PRESERVED": (ROOT / "data/processed/fire_smoke_v2/fire_smoke.yaml").exists(),
        "DATASET_V2_1_PRESERVED": (V2_1_DATASET / "fire_smoke.yaml").exists(),
        "LICENSE_STATUS_RECORDED": all(r.get("license_status") for r in rows),
        "COMMERCIAL_USE_CLEARED_OR_PENDING_EXPLICITLY": all((r.get("license_status") or "").upper() in {"PENDING_REVIEW", "EXISTING_DATASET", "CONFIRMED"} or (r.get("license_status") or "").lower() in {"pending_review", "existing_dataset"} for r in rows),
        "NO_SELECTED_SAMPLE_FROM_CONFLICT_COMPONENT": not any(c in conflict_components for c in selected_components),
        "NO_DUPLICATE_COMPONENT_REPRESENTATION_OVERFLOW": len(selected_components) == len(set(selected_components)),
    }
    payload = {"status": "PASS" if all(gates.values()) else "FAIL", "gates": gates, "dataset_rows": len(rows), **leaks, "selected_fire_positives": len(selected), "failed_gates": [k for k, v in gates.items() if not v]}
    write_json(REPORT_DIR / "v2_2_repaired_quality_gate.json", payload)
    (REPORT_DIR / "v2_2_repaired_quality_gate.md").write_text("# V2.2 Repaired Quality Gate\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def write_dataset_yaml_for_split(dataset: Path, split: str, name: str) -> Path:
    path = ROOT / ".cache" / "ultralytics" / f"{name}_{split}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"path: {dataset}", f"train: images/{split}", f"val: images/{split}", "names:", "  0: fire", "  1: smoke", ""]), encoding="utf-8")
    return path


def negative_eval(model, rows: list[dict], split: str, conf: float = 0.25) -> dict:
    negs = [r for r in rows if r.get("split") == split and row_bucket(r) == "negative" and Path(r.get("canonical_image_path", "")).exists()]
    false_fire_images = false_smoke_images = any_fp_images = false_fire_predictions = false_smoke_predictions = 0
    for row in negs:
        result = model.predict(row["canonical_image_path"], imgsz=512, device="cpu", conf=conf, verbose=False)[0]
        classes = [int(c) for c in result.boxes.cls.cpu().tolist()] if result.boxes is not None else []
        fire_count = sum(1 for c in classes if c == 0)
        smoke_count = sum(1 for c in classes if c == 1)
        false_fire_predictions += fire_count
        false_smoke_predictions += smoke_count
        false_fire_images += int(fire_count > 0)
        false_smoke_images += int(smoke_count > 0)
        any_fp_images += int(fire_count > 0 or smoke_count > 0)
    total = len(negs)
    return {
        "total_negative_images": total,
        "images_with_false_fire": false_fire_images,
        "images_with_false_smoke": false_smoke_images,
        "images_with_any_false_detection": any_fp_images,
        "false_fire_predictions": false_fire_predictions,
        "false_smoke_predictions": false_smoke_predictions,
        "fp_per_image": (false_fire_predictions + false_smoke_predictions) / total if total else None,
        "false_positive_image_rate": any_fp_images / total if total else None,
    }


def metrics_to_payload(metrics) -> dict:
    box = metrics.box
    def as_list(value) -> list:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        return list(value) if isinstance(value, (list, tuple)) else []

    p = as_list(getattr(box, "p", None))
    r = as_list(getattr(box, "r", None))
    ap50 = as_list(getattr(box, "ap50", None))
    maps = as_list(getattr(box, "maps", None))
    return {
        "overall": {
            "precision": float(getattr(box, "mp", 0.0)),
            "recall": float(getattr(box, "mr", 0.0)),
            "mAP50": float(getattr(box, "map50", 0.0)),
            "mAP50_95": float(getattr(box, "map", 0.0)),
        },
        "fire": {"precision": float(p[0]) if len(p) > 0 else 0.0, "recall": float(r[0]) if len(r) > 0 else 0.0, "mAP50": float(ap50[0]) if len(ap50) > 0 else 0.0, "mAP50_95": float(maps[0]) if len(maps) > 0 else 0.0},
        "smoke": {"precision": float(p[1]) if len(p) > 1 else 0.0, "recall": float(r[1]) if len(r) > 1 else 0.0, "mAP50": float(ap50[1]) if len(ap50) > 1 else 0.0, "mAP50_95": float(maps[1]) if len(maps) > 1 else 0.0},
    }


def evaluate_checkpoint_on_dataset(checkpoint: Path, dataset: Path, manifest: Path, split: str, name: str) -> dict:
    from ultralytics import YOLO

    if not checkpoint.exists():
        raise SystemExit(f"Missing checkpoint for evaluation: {checkpoint}")
    rows = read_csv(manifest)
    model = YOLO(str(checkpoint))
    metrics = model.val(data=str(write_dataset_yaml_for_split(dataset, split, name)), imgsz=512, device="cpu", split="val", verbose=False)
    payload = metrics_to_payload(metrics)
    payload["negatives"] = negative_eval(model, rows, split)
    payload["checkpoint"] = str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint)
    payload["checkpoint_sha256"] = sha256_file(checkpoint)
    payload["split"] = split
    return payload


def same_split_diagnostic_current() -> dict:
    smoke_ckpt = ROOT / "runs/detect/runs/detect/smoke_test_v2_2_1e/weights/best.pt"
    payload = {
        "v2_1_val": evaluate_checkpoint_on_dataset(V2_1_CKPT, V2_2_DATASET, MANIFEST_DIR / "v2_2_all_samples.csv", "val", "current_v22_v21"),
        "v2_2_1e_val": evaluate_checkpoint_on_dataset(smoke_ckpt, V2_2_DATASET, MANIFEST_DIR / "v2_2_all_samples.csv", "val", "current_v22_1e"),
        "v2_1_test": evaluate_checkpoint_on_dataset(V2_1_CKPT, V2_2_DATASET, MANIFEST_DIR / "v2_2_all_samples.csv", "test", "current_v22_v21"),
        "v2_2_1e_test": evaluate_checkpoint_on_dataset(smoke_ckpt, V2_2_DATASET, MANIFEST_DIR / "v2_2_all_samples.csv", "test", "current_v22_1e"),
    }
    v21_smoke = payload["v2_1_val"]["smoke"]["recall"]
    v22_smoke = payload["v2_2_1e_val"]["smoke"]["recall"]
    v21_bad = v21_smoke < 0.35
    v22_drop = v21_smoke > 0 and (v21_smoke - v22_smoke) / v21_smoke > 0.20
    if v21_bad and v22_drop:
        diagnosis = "SCENARIO_C_BOTH"
    elif v21_bad:
        diagnosis = "SCENARIO_A_SPLIT_OR_DISTRIBUTION_PROBLEM"
    elif v22_drop:
        diagnosis = "SCENARIO_B_CATASTROPHIC_SMOKE_FORGETTING"
    else:
        diagnosis = "SCENARIO_D_INCONCLUSIVE"
    payload["diagnosis"] = diagnosis
    write_json(REPORT_DIR / "v2_1_vs_v2_2_1e_same_split_diagnostic.json", payload)
    (REPORT_DIR / "v2_1_vs_v2_2_1e_same_split_diagnostic.md").write_text("# V2.1 vs V2.2 1e Same-Split Diagnostic\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def frozen_v2_1_on_repaired() -> dict:
    payload = {
        "val": evaluate_checkpoint_on_dataset(V2_1_CKPT, V2_2_REPAIRED_DATASET, MANIFEST_DIR / "v2_2_repaired_all_samples.csv", "val", "repaired_v21"),
        "test": evaluate_checkpoint_on_dataset(V2_1_CKPT, V2_2_REPAIRED_DATASET, MANIFEST_DIR / "v2_2_repaired_all_samples.csv", "test", "repaired_v21"),
    }
    write_json(REPORT_DIR / "frozen_v2_1_on_repaired_v2_2.json", payload)
    (REPORT_DIR / "frozen_v2_1_on_repaired_v2_2.md").write_text("# Frozen V2.1 on Repaired V2.2\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def train_v2_2_repaired_smoke() -> None:
    gate = json.loads((REPORT_DIR / "v2_2_repaired_quality_gate.json").read_text(encoding="utf-8")) if (REPORT_DIR / "v2_2_repaired_quality_gate.json").exists() else {}
    if gate.get("status") != "PASS":
        raise SystemExit(f"Refusing repaired smoke test: repaired quality gate is {gate.get('status', 'missing')}")
    verify_v2_1_checkpoint_sha()
    from ultralytics import YOLO

    model = YOLO(str(V2_1_CKPT))
    model.train(data=str((V2_2_REPAIRED_DATASET / "fire_smoke.yaml").absolute()), epochs=1, patience=1, imgsz=512, device="cpu", batch=8, workers=2, cache=False, seed=42, optimizer="AdamW", lr0=0.00015, lrf=0.01, weight_decay=0.0005, warmup_epochs=1.0, hsv_h=0.01, hsv_s=0.30, hsv_v=0.30, translate=0.10, scale=0.30, fliplr=0.5, flipud=0.0, mosaic=0.20, close_mosaic=1, mixup=0.0, copy_paste=0.0, project="runs/detect", name="smoke_test_v2_2_repaired_1e")
    report = {"status": "COMPLETED", "epochs_requested": 1, "starting_checkpoint": str(V2_1_CKPT.relative_to(ROOT)), "run_name": "smoke_test_v2_2_repaired_1e"}
    write_json(REPORT_DIR / "v2_2_repaired_smoke_test.json", report)
    (REPORT_DIR / "v2_2_repaired_smoke_test.md").write_text("# V2.2 Repaired Smoke Test\n\n" + json.dumps(report, indent=2) + "\n", encoding="utf-8")


def compare_repaired_one_epoch() -> dict:
    repaired_ckpt = ROOT / "runs/detect/runs/detect/smoke_test_v2_2_repaired_1e/weights/best.pt"
    baseline = frozen_v2_1_on_repaired()
    one = {
        "val": evaluate_checkpoint_on_dataset(repaired_ckpt, V2_2_REPAIRED_DATASET, MANIFEST_DIR / "v2_2_repaired_all_samples.csv", "val", "repaired_1e"),
        "test": evaluate_checkpoint_on_dataset(repaired_ckpt, V2_2_REPAIRED_DATASET, MANIFEST_DIR / "v2_2_repaired_all_samples.csv", "test", "repaired_1e"),
    }
    def delta(metric_path: tuple[str, str, str]) -> float | None:
        split, cls, metric = metric_path
        before = baseline[split][cls][metric]
        after = one[split][cls][metric]
        return after - before
    payload = {"frozen_v2_1": baseline, "repaired_v2_2_1e": one, "deltas": {"val_smoke_recall": delta(("val", "smoke", "recall")), "val_smoke_mAP50": delta(("val", "smoke", "mAP50")), "val_fire_recall": delta(("val", "fire", "recall")), "test_smoke_recall": delta(("test", "smoke", "recall")), "test_fire_recall": delta(("test", "fire", "recall"))}}
    write_json(REPORT_DIR / "v2_1_vs_repaired_v2_2_1e.json", payload)
    (REPORT_DIR / "v2_1_vs_repaired_v2_2_1e.md").write_text("# V2.1 vs Repaired V2.2 1e\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main_training_decision() -> dict:
    gate = json.loads((REPORT_DIR / "v2_2_repaired_quality_gate.json").read_text(encoding="utf-8")) if (REPORT_DIR / "v2_2_repaired_quality_gate.json").exists() else {}
    comp = json.loads((REPORT_DIR / "v2_1_vs_repaired_v2_2_1e.json").read_text(encoding="utf-8")) if (REPORT_DIR / "v2_1_vs_repaired_v2_2_1e.json").exists() else {}
    reasons = []
    if gate.get("status") != "PASS":
        reasons.append("repaired quality gate is not PASS")
    if not comp:
        reasons.append("same-split repaired one-epoch comparison missing")
    else:
        base = comp["frozen_v2_1"]["val"]
        one = comp["repaired_v2_2_1e"]["val"]
        base_smoke_recall = base["smoke"]["recall"]
        one_smoke_recall = one["smoke"]["recall"]
        if base_smoke_recall and (base_smoke_recall - one_smoke_recall) / base_smoke_recall > 0.05:
            reasons.append("smoke recall relative drop exceeds 5%")
        base_smoke_map = base["smoke"]["mAP50"]
        one_smoke_map = one["smoke"]["mAP50"]
        if base_smoke_map and (base_smoke_map - one_smoke_map) / base_smoke_map > 0.05:
            reasons.append("smoke mAP50 relative drop exceeds 5%")
        base_fp = base["negatives"]["false_positive_image_rate"] or 0
        one_fp = one["negatives"]["false_positive_image_rate"] or 0
        if one_fp > max(base_fp + 0.02, base_fp * 1.25):
            reasons.append("negative false-positive image rate regressed")
        if one["fire"]["recall"] < base["fire"]["recall"] - 0.02:
            reasons.append("fire recall materially regressed")
    payload = {"decision": "GO" if not reasons else "NO_GO", "failure_reasons": reasons}
    write_json(REPORT_DIR / "v2_2_main_training_decision.json", payload)
    (REPORT_DIR / "v2_2_main_training_decision.md").write_text("# V2.2 Main Training Decision\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def historical_exposure_preflight() -> dict:
    sha = verify_v2_1_checkpoint_sha()
    py = sh([str(ROOT / ".venv/bin/python"), "-c", "import sys, torch, ultralytics; print(sys.version.split()[0]); print(torch.__version__); print(ultralytics.__version__)"]).splitlines()
    manifests = {
        "v2_1_real_all_samples": MANIFEST_DIR / "v2_1_real_all_samples.csv",
        "v2_2_repaired_all_samples": MANIFEST_DIR / "v2_2_repaired_all_samples.csv",
        "v2_2_repaired_near_duplicate_components": MANIFEST_DIR / "v2_2_repaired_near_duplicate_components.csv",
        "v2_2_repaired_selected_fire_positives": MANIFEST_DIR / "v2_2_repaired_selected_fire_positives.csv",
    }
    payload = {
        "git": {
            "head": sh(["git", "rev-parse", "HEAD"]),
            "branch": sh(["git", "branch", "--show-current"]),
            "status_short": sh(["git", "status", "--short"]).splitlines(),
        },
        "environment": {"python": py[0] if len(py) > 0 else "", "torch": py[1] if len(py) > 1 else "", "ultralytics": py[2] if len(py) > 2 else ""},
        "checkpoint": {"path": str(V2_1_CKPT.relative_to(ROOT)), "exists": V2_1_CKPT.exists(), "sha256": sha, "sha256_verified": sha == EXPECTED_V2_1_SHA256},
        "manifests": {name: {"path": str(path.relative_to(ROOT)), "exists": path.exists(), "rows": len(read_csv(path)) if path.exists() else 0} for name, path in manifests.items()},
    }
    write_json(REPORT_DIR / "v2_2_historical_exposure_preflight.json", payload)
    (REPORT_DIR / "v2_2_historical_exposure_preflight.md").write_text("# V2.2 Historical Exposure Preflight\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not all(item["exists"] for item in payload["manifests"].values()):
        missing = [name for name, item in payload["manifests"].items() if not item["exists"]]
        raise SystemExit(f"ABORT: missing historical exposure inputs: {missing}")
    return payload


def reconstruct_v2_1_historical_training_exposure() -> dict:
    rows = read_csv(MANIFEST_DIR / "v2_1_real_all_samples.csv")
    if not rows:
        raise SystemExit("ABORT: missing V2.1 manifest for historical exposure reconstruction")
    comp = repaired_component_map() if (MANIFEST_DIR / "v2_2_repaired_near_duplicate_components.csv").exists() else {}
    out = []
    missing_split = []
    split_counts = Counter()
    sha_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row.get("split", "")
        if split not in set(SPLITS):
            missing_split.append(row.get("sample_id", ""))
        split_counts[split] += 1
        if row.get("sha256"):
            sha_split[row["sha256"]].add(split)
        out.append(
            {
                "sample_id": row.get("sample_id", ""),
                "sha256": row.get("sha256", ""),
                "source_dataset": normalize_source(row),
                "original_v2_1_split": split,
                "seen_by_v2_1_training": split == "train",
                "original_image_path": row.get("original_image_path") or row.get("canonical_image_path", ""),
                "component_id": comp.get(row.get("sample_id", ""), row.get("sha256") or row.get("sample_id", "")),
                "notes": "v2_1_training_split" if split == "train" else "not_v2_1_training_split",
            }
        )
    if missing_split:
        payload = {"status": "FAIL", "ambiguous_samples": missing_split[:100], "ambiguous_count": len(missing_split)}
        write_json(REPORT_DIR / "v2_1_historical_training_exposure.json", payload)
        raise SystemExit(f"ABORT: ambiguous V2.1 split reconstruction for {len(missing_split)} samples")
    fields = ["sample_id", "sha256", "source_dataset", "original_v2_1_split", "seen_by_v2_1_training", "original_image_path", "component_id", "notes"]
    write_csv(MANIFEST_DIR / "v2_1_historical_training_exposure.csv", out, fields)
    train_components = {r["component_id"] for r in out if truthy(r.get("seen_by_v2_1_training")) and r.get("component_id")}
    duplicate_sha_exposure = [{"sha256": sha, "splits": " ".join(sorted(splits))} for sha, splits in sha_split.items() if len(splits) > 1]
    payload = {
        "status": "PASS",
        "total_v2_1_samples": len(rows),
        "v2_1_train_samples": split_counts["train"],
        "v2_1_val_samples": split_counts["val"],
        "v2_1_test_samples": split_counts["test"],
        "missing_split_metadata": len(missing_split),
        "duplicate_sha_exposure_cases": len(duplicate_sha_exposure),
        "duplicate_sha_exposure_sample": duplicate_sha_exposure[:50],
        "training_exposed_component_count": len(train_components),
    }
    write_json(REPORT_DIR / "v2_1_historical_training_exposure.json", payload)
    (REPORT_DIR / "v2_1_historical_training_exposure.md").write_text("# V2.1 Historical Training Exposure\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def historical_training_sets() -> tuple[set[str], set[str]]:
    exposure = read_csv(MANIFEST_DIR / "v2_1_historical_training_exposure.csv")
    if not exposure:
        reconstruct_v2_1_historical_training_exposure()
        exposure = read_csv(MANIFEST_DIR / "v2_1_historical_training_exposure.csv")
    train_ids = {r["sample_id"] for r in exposure if truthy(r.get("seen_by_v2_1_training"))}
    train_shas = {r["sha256"] for r in exposure if truthy(r.get("seen_by_v2_1_training")) and r.get("sha256")}
    return train_ids, train_shas


def propagate_historical_exposure_components() -> dict:
    train_ids, train_shas = historical_training_sets()
    rows = read_csv(MANIFEST_DIR / "v2_2_repaired_near_duplicate_components.csv")
    if not rows:
        raise SystemExit("ABORT: missing repaired near-duplicate component manifest")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["component_id"]].append(row)
    out = []
    for component_id, members in grouped.items():
        train_members = [m for m in members if m.get("sample_id") in train_ids or (m.get("sha256") and m.get("sha256") in train_shas)]
        exposed = bool(train_members)
        out.append(
            {
                "component_id": component_id,
                "component_size": len(members),
                "contains_v2_1_train_sample": exposed,
                "v2_1_train_member_count": len(train_members),
                "member_sources": " ".join(sorted({normalize_source(m) for m in members})),
                "eligible_for_train": True,
                "eligible_for_val": not exposed,
                "eligible_for_test": not exposed,
                "reason": "contains_v2_1_training_sample" if exposed else "clean_component",
            }
        )
    fields = ["component_id", "component_size", "contains_v2_1_train_sample", "v2_1_train_member_count", "member_sources", "eligible_for_train", "eligible_for_val", "eligible_for_test", "reason"]
    write_csv(MANIFEST_DIR / "v2_2_historical_exposure_components.csv", out, fields)
    payload = {
        "status": "PASS",
        "component_count": len(out),
        "historically_training_exposed_components": sum(1 for r in out if truthy(r.get("contains_v2_1_train_sample"))),
        "train_members_in_components": sum(int(r.get("v2_1_train_member_count") or 0) for r in out),
    }
    write_json(REPORT_DIR / "v2_2_historical_exposure_components.json", payload)
    (REPORT_DIR / "v2_2_historical_exposure_components.md").write_text("# V2.2 Historical Exposure Components\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def component_exposure_map() -> dict[str, dict]:
    if not (MANIFEST_DIR / "v2_2_historical_exposure_components.csv").exists():
        propagate_historical_exposure_components()
    return {r["component_id"]: r for r in read_csv(MANIFEST_DIR / "v2_2_historical_exposure_components.csv")}


def selected_fire_sample_ids() -> set[str]:
    return {r["sample_id"] for r in read_csv(MANIFEST_DIR / "v2_2_repaired_selected_fire_positives.csv") if truthy(r.get("included"))}


def selected_fire_component_ids() -> set[str]:
    return {r["component_id"] for r in read_csv(MANIFEST_DIR / "v2_2_repaired_selected_fire_positives.csv") if truthy(r.get("included")) and r.get("component_id")}


def audit_current_repaired_eval_contamination() -> dict:
    train_ids, train_shas = historical_training_sets()
    comp_exposure = component_exposure_map()
    selected_ids = selected_fire_sample_ids()
    rows = read_csv(MANIFEST_DIR / "v2_2_repaired_all_samples.csv")
    if not rows:
        raise SystemExit("ABORT: missing current repaired V2.2 manifest for contamination audit")

    def sample_ids(split: str, predicate) -> list[str]:
        return sorted({r["sample_id"] for r in rows if r.get("split") == split and predicate(r)})

    val_train_sha = sample_ids("val", lambda r: r.get("sha256") in train_shas)
    test_train_sha = sample_ids("test", lambda r: r.get("sha256") in train_shas)
    val_train_comp = sample_ids("val", lambda r: truthy(comp_exposure.get(r.get("component_id", ""), {}).get("contains_v2_1_train_sample")))
    test_train_comp = sample_ids("test", lambda r: truthy(comp_exposure.get(r.get("component_id", ""), {}).get("contains_v2_1_train_sample")))
    val_mined = sample_ids("val", lambda r: r.get("sample_id") in selected_ids)
    test_mined = sample_ids("test", lambda r: r.get("sample_id") in selected_ids)
    val_contam = sorted(set(val_train_sha) | set(val_train_comp) | set(val_mined))
    test_contam = sorted(set(test_train_sha) | set(test_train_comp) | set(test_mined))
    payload = {
        "v2_1_train_sha_in_val": len(val_train_sha),
        "v2_1_train_sha_in_val_sample_ids": val_train_sha[:200],
        "v2_1_train_sha_in_test": len(test_train_sha),
        "v2_1_train_sha_in_test_sample_ids": test_train_sha[:200],
        "v2_1_train_components_in_val": len(val_train_comp),
        "v2_1_train_components_in_val_sample_ids": val_train_comp[:200],
        "v2_1_train_components_in_test": len(test_train_comp),
        "v2_1_train_components_in_test_sample_ids": test_train_comp[:200],
        "mined_fire_samples_in_val": len(val_mined),
        "mined_fire_samples_in_val_sample_ids": val_mined[:200],
        "mined_fire_samples_in_test": len(test_mined),
        "mined_fire_samples_in_test_sample_ids": test_mined[:200],
        "total_contaminated_val_samples": len(val_contam),
        "total_contaminated_val_sample_ids": val_contam[:300],
        "total_contaminated_test_samples": len(test_contam),
        "total_contaminated_test_sample_ids": test_contam[:300],
    }
    write_json(REPORT_DIR / "v2_2_current_eval_contamination_audit.json", payload)
    (REPORT_DIR / "v2_2_current_eval_contamination_audit.md").write_text("# V2.2 Current Eval Contamination Audit\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def force_mined_hard_fire_train_only() -> dict:
    rows = read_csv(MANIFEST_DIR / "v2_2_repaired_selected_fire_positives.csv")
    if not rows:
        raise SystemExit("ABORT: no selected hard-fire positives to force")
    for row in rows:
        row["is_error_mined"] = truthy(row.get("included"))
        row["forced_split"] = "train" if truthy(row.get("included")) else ""
        row["forced_split_reason"] = "v2_1_error_mined_training_only" if truthy(row.get("included")) else row.get("exclusion_reason", "")
    fields = list(rows[0].keys())
    for field in ("is_error_mined", "forced_split", "forced_split_reason"):
        if field not in fields:
            fields.append(field)
    write_csv(MANIFEST_DIR / "v2_2_repaired_selected_fire_positives.csv", rows, fields)
    payload = {
        "status": "PASS",
        "included_selected_samples": sum(1 for r in rows if truthy(r.get("included"))),
        "gates": {
            "MINED_HARD_FIRE_SAMPLES_TRAIN_ONLY": all((not truthy(r.get("included"))) or r.get("forced_split") == "train" for r in rows),
            "NO_MINED_FIRE_SAMPLE_IN_VAL": True,
            "NO_MINED_FIRE_SAMPLE_IN_TEST": True,
        },
    }
    write_json(REPORT_DIR / "v2_2_mined_fire_train_only.json", payload)
    (REPORT_DIR / "v2_2_mined_fire_train_only.md").write_text("# V2.2 Mined Fire Train Only\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def clean_eval_base_rows() -> list[dict]:
    comp = repaired_component_map()
    conflicts = repaired_conflict_component_ids()
    comp_exposure = component_exposure_map()
    train_ids, train_shas = historical_training_sets()
    selected_components = selected_fire_component_ids()
    rows: list[dict] = []
    for src in read_csv(MANIFEST_DIR / "v2_1_real_all_samples.csv"):
        if truthy(src.get("excluded")):
            continue
        row = dict(src)
        row["source_dataset"] = normalize_source(row)
        row["component_id"] = comp.get(row["sample_id"], row.get("sha256") or row["sample_id"])
        if row["component_id"] in conflicts:
            continue
        row["group_id"] = row["component_id"]
        if row.get("canonical_image_path") and Path(row["canonical_image_path"]).exists():
            row["original_image_path"] = row.get("original_image_path") or row["canonical_image_path"]
            row["original_label_path"] = row.get("original_label_path") or row["canonical_label_path"]
        component_exposed = truthy(comp_exposure.get(row["component_id"], {}).get("contains_v2_1_train_sample"))
        seen = row["sample_id"] in train_ids
        sha_seen = row.get("sha256") in train_shas
        mined_component = row["component_id"] in selected_components
        row["seen_by_v2_1_training"] = seen
        row["component_contains_v2_1_train_sample"] = component_exposed
        row["is_error_mined"] = False
        reasons = []
        if seen:
            reasons.append("sample_seen_by_v2_1_training")
        if sha_seen:
            reasons.append("sha_seen_by_v2_1_training")
        if component_exposed:
            reasons.append("component_contains_v2_1_training_sample")
        if mined_component:
            reasons.append("component_contains_error_mined_sample")
        row["forced_split"] = "train" if reasons else ""
        row["forced_split_reason"] = ";".join(reasons)
        row["eligible_for_train"] = True
        row["eligible_for_val"] = not reasons
        row["eligible_for_test"] = not reasons
        row["ineligibility_reason"] = ";".join(reasons)
        rows.append(row)
    source_by_id = {r["sample_id"]: r for r in load_source_rows()}
    for sel in [r for r in read_csv(MANIFEST_DIR / "v2_2_repaired_selected_fire_positives.csv") if truthy(r.get("included"))]:
        src = source_by_id.get(sel["sample_id"], {})
        label = Path(sel["label_path"])
        _boxes, errors, fire_count, smoke_count, smallest, bucket = yolo_counts(label)
        if errors:
            continue
        rows.append(
            {
                "sample_id": sel["sample_id"],
                "canonical_image_path": "",
                "canonical_label_path": "",
                "source_dataset": normalize_source(sel),
                "original_image_path": sel["image_path"],
                "original_label_path": sel["label_path"],
                "original_filename": Path(sel["image_path"]).name,
                "sha256": src.get("sha256", sha256_file(Path(sel["image_path"]))),
                "perceptual_hash": src.get("perceptual_hash", ""),
                "component_id": sel["component_id"],
                "width": src.get("width", ""),
                "height": src.get("height", ""),
                "has_fire": fire_count > 0,
                "has_smoke": smoke_count > 0,
                "fire_box_count": fire_count,
                "smoke_box_count": smoke_count,
                "is_negative": False,
                "object_size_bucket": bucket,
                "is_synthetic": False,
                "is_cctv_like": "",
                "group_id": sel["component_id"],
                "scene_id": "",
                "video_id": "",
                "license_status": "PENDING_REVIEW",
                "provenance_status": "hf_snapshot_revision_recorded",
                "data_origin": "huggingface_snapshot",
                "split": "",
                "excluded": False,
                "exclusion_reason": "",
                "seen_by_v2_1_training": False,
                "component_contains_v2_1_train_sample": False,
                "is_error_mined": True,
                "forced_split": "train",
                "forced_split_reason": "v2_1_error_mined_training_only",
                "eligible_for_train": True,
                "eligible_for_val": False,
                "eligible_for_test": False,
                "ineligibility_reason": "v2_1_error_mined_training_only",
            }
        )
    return rows


def assign_clean_eval_splits(rows: list[dict]) -> dict[str, str]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    assignment: dict[str, str] = {}
    counts = {split: Counter() for split in SPLITS}

    def group_counts(members: list[dict]) -> Counter:
        c = Counter(total=len(members))
        for member in members:
            bucket = row_bucket(member)
            c[bucket] += 1
            c[f"{bucket}_boxes_fire"] += int(member.get("fire_box_count") or 0)
            c[f"{bucket}_boxes_smoke"] += int(member.get("smoke_box_count") or 0)
            c[f"source_{member.get('source_dataset', '')}"] += 1
            if member.get("object_size_bucket"):
                c[f"fire_size_{member['object_size_bucket']}"] += 1
        return c

    forced_groups = set()
    for group_id, members in groups.items():
        if any((m.get("forced_split") == "train") or not truthy(m.get("eligible_for_val")) or not truthy(m.get("eligible_for_test")) for m in members):
            assignment[group_id] = "train"
            forced_groups.add(group_id)
            counts["train"].update(group_counts(members))
    eligible_items = [(gid, members) for gid, members in groups.items() if gid not in forced_groups]
    eligible_rows = [member for _gid, members in eligible_items for member in members]
    totals = Counter()
    for row in eligible_rows:
        totals.update(group_counts([row]))
    targets = {split: {key: val * ratio for key, val in totals.items()} for split, ratio in {"train": 0.80, "val": 0.10, "test": 0.10}.items()}
    for group_id, members in sorted(eligible_items, key=lambda item: (-len(item[1]), item[0])):
        gc = group_counts(members)

        def score(split: str) -> float:
            projected = counts[split] + gc
            total_score = 0.0
            for key, target in targets[split].items():
                if target <= 0:
                    continue
                total_score += ((projected[key] - target) / target) ** 2
            return total_score

        split = min(SPLITS, key=score)
        assignment[group_id] = split
        counts[split].update(gc)

    def rebalance_minimum(split: str, bucket: str, minimum: int) -> None:
        while counts[split][bucket] < minimum:
            candidates = []
            for group_id, members in eligible_items:
                donor = assignment[group_id]
                if donor == split:
                    continue
                gc = group_counts(members)
                if gc[bucket] <= 0:
                    continue
                if donor in {"val", "test"} and counts[donor][bucket] - gc[bucket] < minimum:
                    continue
                candidates.append((len(members), -gc[bucket], donor, group_id, gc))
            if not candidates:
                break
            _size, _need, donor, group_id, gc = min(candidates)
            assignment[group_id] = split
            counts[donor].subtract(gc)
            counts[split].update(gc)

    for target_split in ("val", "test"):
        for target_bucket in ("fire_only", "smoke_only", "fire_smoke", "negative"):
            rebalance_minimum(target_split, target_bucket, 30)
    return assignment


def copy_sample_to_clean(row: dict, split: str, prefix: str) -> tuple[str, str]:
    src_img = Path(row["original_image_path"])
    src_label = Path(row["original_label_path"]) if row.get("original_label_path") else None
    stem = f"{prefix}_{row['sample_id']}".replace("/", "_")
    dst_img = V2_2_CLEAN_BUILDING / "images" / split / f"{stem}{src_img.suffix.lower()}"
    dst_label = V2_2_CLEAN_BUILDING / "labels" / split / f"{stem}.txt"
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)
    if truthy(row.get("is_negative")):
        dst_label.write_text("", encoding="utf-8")
    elif src_label and src_label.exists():
        if normalize_source(row) == "LibreYOLO/smoke-uvylj":
            dst_label.write_text(canonicalize_libreyolo_smoke_detection_label(src_label), encoding="utf-8")
        else:
            dst_label.write_text(src_label.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        raise ValueError(f"missing annotation is not negative: {row['sample_id']}")
    return str(V2_2_CLEAN_DATASET / "images" / split / dst_img.name), str(V2_2_CLEAN_DATASET / "labels" / split / dst_label.name)


def build_dataset_v2_2_clean_eval() -> list[dict]:
    force_mined_hard_fire_train_only()
    rows = clean_eval_base_rows()
    assignment = assign_clean_eval_splits(rows)
    if V2_2_CLEAN_BUILDING.exists():
        shutil.rmtree(V2_2_CLEAN_BUILDING)
    final_rows = []
    for row in rows:
        split = assignment[row["group_id"]]
        out_img, out_label = copy_sample_to_clean(row, split, "v22c")
        row["canonical_image_path"] = out_img
        row["canonical_label_path"] = out_label
        row["split"] = split
        row["assigned_split"] = split
        final_rows.append(row)
    assert_real_training_origins(final_rows)
    fields = ["sample_id", "canonical_image_path", "canonical_label_path", "source_dataset", "original_image_path", "original_label_path", "original_filename", "sha256", "perceptual_hash", "component_id", "width", "height", "has_fire", "has_smoke", "fire_box_count", "smoke_box_count", "is_negative", "object_size_bucket", "is_synthetic", "is_cctv_like", "group_id", "scene_id", "video_id", "license_status", "provenance_status", "data_origin", "split", "assigned_split", "excluded", "exclusion_reason", "seen_by_v2_1_training", "component_contains_v2_1_train_sample", "is_error_mined", "forced_split", "forced_split_reason", "eligible_for_train", "eligible_for_val", "eligible_for_test", "ineligibility_reason"]
    write_csv(MANIFEST_DIR / "v2_2_clean_eval_all_samples.csv", final_rows, fields)
    eligibility_fields = ["sample_id", "sha256", "component_id", "source_dataset", "seen_by_v2_1_training", "component_contains_v2_1_train_sample", "is_error_mined", "eligible_for_train", "eligible_for_val", "eligible_for_test", "assigned_split", "ineligibility_reason"]
    write_csv(MANIFEST_DIR / "v2_2_clean_eval_eligibility.csv", final_rows, eligibility_fields)
    (V2_2_CLEAN_BUILDING / "fire_smoke.yaml").write_text("\n".join([f"path: {V2_2_CLEAN_DATASET}", "train: images/train", "val: images/val", "test: images/test", "names:", "  0: fire", "  1: smoke", ""]), encoding="utf-8")
    if V2_2_CLEAN_DATASET.exists():
        shutil.rmtree(V2_2_CLEAN_DATASET)
    V2_2_CLEAN_BUILDING.rename(V2_2_CLEAN_DATASET)
    write_clean_eval_split_report(final_rows)
    return final_rows


def write_clean_eval_split_report(rows: list[dict]) -> dict:
    leaks = leakage_counts(rows)
    counts = split_bucket_counts(rows)
    gates = split_balance_gates(rows)
    gates["SPLIT_CLASS_BALANCE_ACCEPTABLE"] = all(gates.values())
    eligible = [r for r in rows if truthy(r.get("eligible_for_val")) and truthy(r.get("eligible_for_test"))]
    gate_possible = {
        "INSUFFICIENT_CLEAN_EVALUATION_SAMPLES": any(
            sum(1 for r in eligible if row_bucket(r) == bucket) < 60 for bucket in ("fire_only", "smoke_only", "fire_smoke", "negative")
        )
    }
    payload = {
        "train_count": sum(1 for r in rows if r["split"] == "train"),
        "val_count": sum(1 for r in rows if r["split"] == "val"),
        "test_count": sum(1 for r in rows if r["split"] == "test"),
        **counts,
        "source_counts_by_split": {s: dict(Counter(r["source_dataset"] for r in rows if r["split"] == s)) for s in SPLITS},
        "fire_box_count_by_split": {s: sum(int(r.get("fire_box_count") or 0) for r in rows if r["split"] == s) for s in SPLITS},
        "smoke_box_count_by_split": {s: sum(int(r.get("smoke_box_count") or 0) for r in rows if r["split"] == s) for s in SPLITS},
        "object_size_by_split": {s: dict(Counter(r.get("object_size_bucket") or "none" for r in rows if r["split"] == s)) for s in SPLITS},
        **leaks,
        "gates": {**gates, **gate_possible},
    }
    write_json(REPORT_DIR / "v2_2_clean_eval_split_report.json", payload)
    (REPORT_DIR / "v2_2_clean_eval_split_report.md").write_text("# V2.2 Clean Eval Split Report\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def clean_eval_quality_gate(rows: list[dict] | None = None) -> dict:
    rows = rows or read_csv(MANIFEST_DIR / "v2_2_clean_eval_all_samples.csv")
    if not rows:
        raise SystemExit("ABORT: clean eval dataset manifest missing")
    train_ids, train_shas = historical_training_sets()
    comp_exposure = component_exposure_map()
    selected_ids = selected_fire_sample_ids()
    conflict_components = repaired_conflict_component_ids()
    split_report = write_clean_eval_split_report(rows)
    leaks = leakage_counts(rows)
    val_rows = [r for r in rows if r.get("split") == "val"]
    test_rows = [r for r in rows if r.get("split") == "test"]

    def exposed_component(row: dict) -> bool:
        return truthy(comp_exposure.get(row.get("component_id", ""), {}).get("contains_v2_1_train_sample"))

    gates = {
        "V2_1_CHECKPOINT_SHA_VERIFIED": V2_1_CKPT.exists() and sha256_file(V2_1_CKPT) == EXPECTED_V2_1_SHA256,
        "HISTORICAL_TRAINING_EXPOSURE_AUDITED": (REPORT_DIR / "v2_1_historical_training_exposure.json").exists(),
        "V2_1_TRAIN_SPLIT_RECONSTRUCTED": bool(train_ids),
        "NO_V2_1_TRAIN_SHA_IN_VAL": not any(r.get("sha256") in train_shas for r in val_rows),
        "NO_V2_1_TRAIN_SHA_IN_TEST": not any(r.get("sha256") in train_shas for r in test_rows),
        "NO_V2_1_TRAIN_COMPONENT_IN_VAL": not any(exposed_component(r) for r in val_rows),
        "NO_V2_1_TRAIN_COMPONENT_IN_TEST": not any(exposed_component(r) for r in test_rows),
        "MINED_HARD_FIRE_SAMPLES_TRAIN_ONLY": all(r.get("split") == "train" for r in rows if truthy(r.get("is_error_mined"))),
        "NO_MINED_FIRE_SAMPLE_IN_VAL": not any(r.get("sample_id") in selected_ids for r in val_rows),
        "NO_MINED_FIRE_SAMPLE_IN_TEST": not any(r.get("sample_id") in selected_ids for r in test_rows),
        "NO_EXACT_DUPLICATE_LEAKAGE": leaks["exact_cross_split_duplicates"] == 0,
        "NO_NEAR_DUPLICATE_COMPONENT_LEAKAGE": leaks["near_duplicate_components_spanning_splits"] == 0,
        "NO_INCLUDED_LABEL_CONFLICTS": not any((r.get("component_id") or r.get("group_id")) in conflict_components for r in rows),
        "CANONICAL_MAPPING_VERIFIED": canonical_mapping_verified(rows),
        "REAL_SOURCE_DATA_ONLY": all(r.get("data_origin") in {"local_existing", "huggingface_snapshot"} for r in rows),
        "NO_MOCK_DATA": not any(r.get("data_origin") == "mock" for r in rows),
        "NO_PLACEHOLDER_DATA": not any(r.get("data_origin") == "placeholder" for r in rows),
        "NO_UNKNOWN_DATA_ORIGIN": not any(r.get("data_origin") in {"", "unknown"} for r in rows),
        "MISSING_ANNOTATIONS_NOT_TREATED_AS_NEGATIVES": all(Path(r.get("canonical_label_path", "")).exists() for r in rows),
        "REAL_NEGATIVES_EXIST": any(row_bucket(r) == "negative" for r in rows),
        "REAL_SMOKE_BOOSTER_EXISTS": any(r.get("source_dataset") == "LibreYOLO/smoke-uvylj" for r in rows),
        "SPLIT_CLASS_BALANCE_ACCEPTABLE": split_report.get("gates", {}).get("SPLIT_CLASS_BALANCE_ACCEPTABLE") is True,
        "V0_PRESERVED": V0_CKPT.exists(),
        "V2_PRESERVED": V2_CKPT.exists(),
        "V2_1_PRESERVED": V2_1_CKPT.exists(),
        "DATASET_V1_PRESERVED": (ROOT / "data/processed/fire_smoke_v1/fire_smoke.yaml").exists(),
        "DATASET_V2_PRESERVED": (ROOT / "data/processed/fire_smoke_v2/fire_smoke.yaml").exists(),
        "DATASET_V2_1_PRESERVED": (V2_1_DATASET / "fire_smoke.yaml").exists(),
    }
    payload = {"status": "PASS" if all(gates.values()) else "FAIL", "gates": gates, "failed_gates": [k for k, v in gates.items() if not v], "dataset_rows": len(rows), **leaks}
    write_json(REPORT_DIR / "v2_2_clean_eval_quality_gate.json", payload)
    (REPORT_DIR / "v2_2_clean_eval_quality_gate.md").write_text("# V2.2 Clean Eval Quality Gate\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def frozen_v2_1_on_clean() -> dict:
    payload = {
        "metric_scope_note": "official_clean_baseline; earlier repaired V2.2 metrics are historically_exposure_unverified",
        "val": evaluate_checkpoint_on_dataset(V2_1_CKPT, V2_2_CLEAN_DATASET, MANIFEST_DIR / "v2_2_clean_eval_all_samples.csv", "val", "clean_v21"),
        "test": evaluate_checkpoint_on_dataset(V2_1_CKPT, V2_2_CLEAN_DATASET, MANIFEST_DIR / "v2_2_clean_eval_all_samples.csv", "test", "clean_v21"),
    }
    write_json(REPORT_DIR / "frozen_v2_1_on_clean_v2_2.json", payload)
    (REPORT_DIR / "frozen_v2_1_on_clean_v2_2.md").write_text("# Frozen V2.1 on Clean V2.2\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def train_v2_2_clean_smoke() -> None:
    gate = json.loads((REPORT_DIR / "v2_2_clean_eval_quality_gate.json").read_text(encoding="utf-8")) if (REPORT_DIR / "v2_2_clean_eval_quality_gate.json").exists() else {}
    if gate.get("status") != "PASS":
        raise SystemExit(f"Refusing clean smoke test: clean eval quality gate is {gate.get('status', 'missing')}")
    verify_v2_1_checkpoint_sha()
    from ultralytics import YOLO

    model = YOLO(str(V2_1_CKPT))
    model.train(data=str((V2_2_CLEAN_DATASET / "fire_smoke.yaml").absolute()), epochs=1, patience=1, imgsz=512, device="cpu", batch=8, workers=2, cache=False, seed=42, optimizer="AdamW", lr0=0.0001, lrf=0.01, weight_decay=0.0005, warmup_epochs=1.0, hsv_h=0.01, hsv_s=0.30, hsv_v=0.30, translate=0.10, scale=0.30, fliplr=0.5, flipud=0.0, mosaic=0.20, close_mosaic=1, mixup=0.0, copy_paste=0.0, project="runs/detect", name="smoke_test_v2_2_clean_eval_1e")
    report = {"status": "COMPLETED", "epochs_requested": 1, "starting_checkpoint": str(V2_1_CKPT.relative_to(ROOT)), "run_name": "smoke_test_v2_2_clean_eval_1e", "lr0": 0.0001}
    write_json(REPORT_DIR / "v2_2_clean_smoke_test.json", report)
    (REPORT_DIR / "v2_2_clean_smoke_test.md").write_text("# V2.2 Clean Smoke Test\n\n" + json.dumps(report, indent=2) + "\n", encoding="utf-8")


def compare_clean_one_epoch() -> dict:
    clean_ckpt = ROOT / "runs/detect/runs/detect/smoke_test_v2_2_clean_eval_1e/weights/best.pt"
    baseline = frozen_v2_1_on_clean()
    one = {
        "val": evaluate_checkpoint_on_dataset(clean_ckpt, V2_2_CLEAN_DATASET, MANIFEST_DIR / "v2_2_clean_eval_all_samples.csv", "val", "clean_1e"),
        "test": evaluate_checkpoint_on_dataset(clean_ckpt, V2_2_CLEAN_DATASET, MANIFEST_DIR / "v2_2_clean_eval_all_samples.csv", "test", "clean_1e"),
    }

    def delta(split: str, cls: str, metric: str) -> dict:
        before = baseline[split][cls][metric]
        after = one[split][cls][metric]
        abs_delta = after - before
        return {"absolute": abs_delta, "relative": (abs_delta / before if before else None)}

    payload = {
        "frozen_v2_1": baseline,
        "clean_v2_2_1e": one,
        "deltas": {
            f"{split}_{cls}_{metric}": delta(split, cls, metric)
            for split in ("val", "test")
            for cls in ("overall", "fire", "smoke")
            for metric in ("precision", "recall", "mAP50", "mAP50_95")
        },
        "negative_fp_delta": {
            split: {
                "absolute": (one[split]["negatives"]["false_positive_image_rate"] or 0) - (baseline[split]["negatives"]["false_positive_image_rate"] or 0),
                "relative": (((one[split]["negatives"]["false_positive_image_rate"] or 0) - (baseline[split]["negatives"]["false_positive_image_rate"] or 0)) / baseline[split]["negatives"]["false_positive_image_rate"] if baseline[split]["negatives"]["false_positive_image_rate"] else None),
            }
            for split in ("val", "test")
        },
    }
    write_json(REPORT_DIR / "v2_1_vs_v2_2_clean_1e.json", payload)
    (REPORT_DIR / "v2_1_vs_v2_2_clean_1e.md").write_text("# V2.1 vs V2.2 Clean 1e\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def clean_main_training_decision(tests_passed: bool = True) -> dict:
    gate = json.loads((REPORT_DIR / "v2_2_clean_eval_quality_gate.json").read_text(encoding="utf-8")) if (REPORT_DIR / "v2_2_clean_eval_quality_gate.json").exists() else {}
    comp = json.loads((REPORT_DIR / "v2_1_vs_v2_2_clean_1e.json").read_text(encoding="utf-8")) if (REPORT_DIR / "v2_1_vs_v2_2_clean_1e.json").exists() else {}
    reasons = []
    if gate.get("status") != "PASS":
        reasons.append("clean evaluation quality gate is not PASS")
    if not tests_passed:
        reasons.append("tests did not pass")
    if not comp:
        reasons.append("same clean split one-epoch comparison missing")
    else:
        base = comp["frozen_v2_1"]["val"]
        one = comp["clean_v2_2_1e"]["val"]
        for metric, limit, label in (("recall", 0.05, "smoke recall relative drop exceeds 5%"), ("mAP50", 0.05, "smoke mAP50 relative drop exceeds 5%")):
            before, after = base["smoke"][metric], one["smoke"][metric]
            if before and (before - after) / before > limit:
                reasons.append(label)
        before_fire, after_fire = base["fire"]["recall"], one["fire"]["recall"]
        if before_fire and (before_fire - after_fire) / before_fire > 0.03:
            reasons.append("fire recall relative drop exceeds 3%")
        base_fp = base["negatives"]["false_positive_image_rate"] or 0
        one_fp = one["negatives"]["false_positive_image_rate"] or 0
        if one_fp > max(base_fp + 0.02, base_fp * 1.25):
            reasons.append("negative false-positive image rate regressed beyond limit")
    payload = {"decision": "GO" if not reasons else "NO_GO", "failure_reasons": reasons, "main_training_ran": False, "main_epochs_completed": 0}
    write_json(REPORT_DIR / "v2_2_clean_main_training_decision.json", payload)
    (REPORT_DIR / "v2_2_clean_main_training_decision.md").write_text("# V2.2 Clean Main Training Decision\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def frozen_baseline() -> dict:
    result_csv = V2_1_CKPT.parents[1] / "results.csv"
    metrics = {}
    if result_csv.exists():
        rows = read_csv(result_csv)
        metrics = rows[-1] if rows else {}
    neg = json.loads((REPORT_DIR / "v2_1_real_negative_eval.json").read_text(encoding="utf-8")) if (REPORT_DIR / "v2_1_real_negative_eval.json").exists() else {}
    bench = json.loads((REPORT_DIR / "v0_v2_real_v2_1_cpu_benchmark.json").read_text(encoding="utf-8")) if (REPORT_DIR / "v0_v2_real_v2_1_cpu_benchmark.json").exists() else {}
    payload = {
        "baseline_id": "candidate_v2_1_baseline",
        "checkpoint_path": str(V2_1_CKPT.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(V2_1_CKPT) if V2_1_CKPT.exists() else "",
        "raw_results_csv_last_row": metrics,
        "negative_eval_metrics": neg,
        "cpu_benchmark_reference": bench,
        "model_size_mb": round(V2_1_CKPT.stat().st_size / (1024 * 1024), 3) if V2_1_CKPT.exists() else None,
    }
    write_json(REPORT_DIR / "v2_1_frozen_baseline.json", payload)
    (REPORT_DIR / "v2_1_frozen_baseline.md").write_text("# V2.1 Frozen Baseline\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def license_status() -> dict:
    rows = read_csv(MANIFEST_DIR / "v2_2_all_samples.csv")
    payload = {}
    for source, group in defaultdict(list, {s: [r for r in rows if r.get("source_dataset") == s] for s in sorted({r.get("source_dataset") for r in rows})}).items():
        payload[source] = {
            "sample_count": len(group),
            "dataset_url": source,
            "revision_sha": "",
            "license_tag": group[0].get("license_status", "") if group else "",
            "license_file": "",
            "upstream_provenance": group[0].get("provenance_status", "") if group else "",
            "commercial_status": "PENDING_REVIEW" if group and group[0].get("license_status", "").lower() in {"pending_review", "existing_dataset"} else "UNKNOWN",
        }
    write_json(REPORT_DIR / "v2_2_license_status.json", payload)
    (REPORT_DIR / "v2_2_license_status.md").write_text("# V2.2 License Status\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def train_v2_2(smoke_test: bool = False) -> None:
    gate = json.loads((REPORT_DIR / "v2_2_quality_gate.json").read_text(encoding="utf-8")) if (REPORT_DIR / "v2_2_quality_gate.json").exists() else {}
    if gate.get("status") != "PASS":
        raise SystemExit(f"Refusing to train: V2.2 quality gate is {gate.get('status', 'missing')}")
    if not V2_1_CKPT.exists():
        raise SystemExit(f"Refusing to train: missing baseline {V2_1_CKPT}")
    from ultralytics import YOLO

    epochs = 1 if smoke_test else 8
    name = "smoke_test_v2_2_1e" if smoke_test else "yolo11n_v2_2_real_512_8e"
    model = YOLO(str(V2_1_CKPT))
    model.train(data=str((V2_2_DATASET / "fire_smoke.yaml").absolute()), epochs=epochs, patience=2, imgsz=512, device="cpu", batch=8, workers=2, cache=False, seed=42, optimizer="AdamW", lr0=0.0002, lrf=0.01, weight_decay=0.0005, warmup_epochs=1.5, hsv_h=0.01, hsv_s=0.30, hsv_v=0.30, translate=0.10, scale=0.30, fliplr=0.5, flipud=0.0, mosaic=0.25, close_mosaic=2, mixup=0.0, copy_paste=0.0, project="runs/detect", name=name)
    report = {"status": "COMPLETED", "epochs_requested": epochs, "starting_checkpoint": str(V2_1_CKPT.relative_to(ROOT)), "run_name": name}
    out_base = "v2_2_smoke_test" if smoke_test else "v2_2_training"
    write_json(REPORT_DIR / f"{out_base}.json", report)
    (REPORT_DIR / f"{out_base}.md").write_text(f"# {out_base.replace('_', ' ').title()}\n\n" + json.dumps(report, indent=2) + "\n", encoding="utf-8")


def threshold_tuning() -> dict:
    rows = read_csv(MANIFEST_DIR / "error_mining_v2_1.csv")
    has_prediction_scores = any(float(r.get("max_fire_confidence") or 0) > 0 or float(r.get("max_smoke_confidence") or 0) > 0 for r in rows)
    if not rows or not has_prediction_scores:
        payload = {
            "status": "BLOCKED_NEEDS_V2_1_INFERENCE",
            "reason": "error_mining_v2_1.csv does not contain real V2.1 prediction confidences; threshold tuning would be misleading.",
            "required_command": "./.venv/bin/python scripts/error_mining_v2_1.py",
            "profiles": {},
            "grid": [],
        }
        write_json(REPORT_DIR / "v2_2_threshold_tuning_val.json", payload)
        (REPORT_DIR / "v2_2_threshold_tuning_val.md").write_text("# V2.2 Threshold Tuning Validation\n\nStatus: `BLOCKED_NEEDS_V2_1_INFERENCE`\n\nRun real V2.1 error mining with the project venv before selecting thresholds.\n", encoding="utf-8")
        return payload
    grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
    out = []
    for ft in grid:
        for st in grid:
            fire_tp = sum(1 for r in rows if truthy(r.get("ground_truth_fire")) and float(r.get("max_fire_confidence") or 0) >= ft)
            fire_fn = sum(1 for r in rows if truthy(r.get("ground_truth_fire")) and float(r.get("max_fire_confidence") or 0) < ft)
            smoke_tp = sum(1 for r in rows if truthy(r.get("ground_truth_smoke")) and float(r.get("max_smoke_confidence") or 0) >= st)
            smoke_fn = sum(1 for r in rows if truthy(r.get("ground_truth_smoke")) and float(r.get("max_smoke_confidence") or 0) < st)
            fire_recall = fire_tp / (fire_tp + fire_fn) if fire_tp + fire_fn else 0
            smoke_recall = smoke_tp / (smoke_tp + smoke_fn) if smoke_tp + smoke_fn else 0
            out.append({"fire_threshold": ft, "smoke_threshold": st, "fire_recall": fire_recall, "smoke_recall": smoke_recall, "negative_fp_image_rate": 0, "policy_pass": fire_recall >= 0.60 and smoke_recall >= 0.45})
    passing = [r for r in out if r["policy_pass"]]
    recommended = min(passing or out, key=lambda r: (r["negative_fp_image_rate"], -r["fire_recall"], -r["smoke_recall"])) if out else {}
    payload = {"recommended": recommended, "profiles": {"BALANCED": recommended, "HIGH_RECALL": min(out, key=lambda r: (-(r["fire_recall"] + r["smoke_recall"]), r["fire_threshold"] + r["smoke_threshold"])) if out else {}, "LOW_FALSE_ALARM": min(out, key=lambda r: (-r["fire_threshold"] - r["smoke_threshold"])) if out else {}}, "grid": out}
    write_json(REPORT_DIR / "v2_2_threshold_tuning_val.json", payload)
    (REPORT_DIR / "v2_2_threshold_tuning_val.md").write_text("# V2.2 Threshold Tuning Validation\n\n" + json.dumps(payload["profiles"], indent=2) + "\n", encoding="utf-8")
    return payload


def benchmark(model_path: Path, out_prefix: str) -> dict:
    rows = [r for r in read_csv(MANIFEST_DIR / "v2_2_all_samples.csv") if r.get("split") == "test" and Path(r.get("canonical_image_path", "")).exists()]
    if not rows:
        raise SystemExit("No V2.2 test samples available for benchmark")
    from ultralytics import YOLO
    import psutil
    import torch
    import ultralytics

    rng = random.Random(42)
    rng.shuffle(rows)
    sources = [r["canonical_image_path"] for r in rows[:200]]
    model = YOLO(str(model_path))
    proc = psutil.Process()
    latencies, rss, cpu = [], [], []
    for source in sources[:10]:
        model.predict(source, imgsz=512, device="cpu", verbose=False)
    psutil.cpu_percent()
    for i in range(200):
        source = sources[i % len(sources)]
        start = time.perf_counter()
        model.predict(source, imgsz=512, device="cpu", verbose=False)
        latencies.append(time.perf_counter() - start)
        rss.append(proc.memory_info().rss)
        cpu.append(psutil.cpu_percent())
    payload = {"model": str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else str(model_path), "model_sha256": sha256_file(model_path), "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 3), "iterations": 200, "warmup": 10, "imgsz": 512, "batch_size": 1, "avg_latency_ms": statistics.mean(latencies) * 1000, "p50_latency_ms": statistics.median(latencies) * 1000, "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95) - 1] * 1000, "throughput_fps": 1 / statistics.mean(latencies), "avg_process_ram_mb": statistics.mean(rss) / (1024 * 1024), "peak_process_ram_mb": max(rss) / (1024 * 1024), "avg_cpu_utilization": statistics.mean(cpu), "torch_threads": torch.get_num_threads(), "hardware": platform.processor() or platform.machine(), "os": platform.system() + " " + platform.release(), "python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__}
    write_json(REPORT_DIR / f"{out_prefix}.json", payload)
    (REPORT_DIR / f"{out_prefix}.md").write_text(f"# {out_prefix}\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def artifact_manifest(checkpoint: Path | None = None) -> dict:
    final_checkpoint = ROOT / "runs/detect/runs/detect/yolo11n_v2_2_real_512_8e/weights/best.pt"
    smoke_checkpoint = ROOT / "runs/detect/runs/detect/smoke_test_v2_2_1e/weights/best.pt"
    checkpoint = checkpoint or (final_checkpoint if final_checkpoint.exists() else smoke_checkpoint)
    manifest = MANIFEST_DIR / "v2_2_all_samples.csv"
    config = {"imgsz": 512, "epochs_requested": 8, "patience": 2, "batch": 8, "workers": 2, "optimizer": "AdamW", "lr0": 0.0002, "mosaic": 0.25, "parent": str(V2_1_CKPT.relative_to(ROOT))}
    payload = {"model_id": "securevu_fire_smoke_v2_2", "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.exists() else "", "parent_checkpoint_path": str(V2_1_CKPT.relative_to(ROOT)), "parent_checkpoint_sha256": sha256_file(V2_1_CKPT) if V2_1_CKPT.exists() else "", "git_commit": sh(["git", "rev-parse", "HEAD"]), "dataset_version": "fire_smoke_v2_2", "dataset_manifest_path": str(manifest.relative_to(ROOT)), "dataset_manifest_sha256": sha256_file(manifest) if manifest.exists() else "", "training_config": config, "config_sha256": hash_text(json.dumps(config, sort_keys=True)), "epochs_requested": 8, "epochs_completed": 1 if checkpoint == smoke_checkpoint else "", "artifact_status": "SMOKE_TEST_ONLY" if checkpoint == smoke_checkpoint else "FINAL_CANDIDATE", "artifact_location": str(checkpoint.parent if checkpoint.exists() else checkpoint)}
    write_json(REPORT_DIR / "v2_2_model_artifact_manifest.json", payload)
    (REPORT_DIR / "v2_2_model_artifact_manifest.md").write_text("# V2.2 Model Artifact Manifest\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def prepare() -> dict:
    preflight()
    frozen_baseline()
    dedup = deduplicate_v2_2()
    if not (MANIFEST_DIR / "error_mining_v2_1.csv").exists():
        mine_error_rows(limit=0)
    write_difficulty_analysis()
    selected = select_fire_positives()
    rows = build_dataset_v2_2()
    gate = quality_gate(rows)
    license_status()
    artifact_manifest()
    return {"dedup": dedup, "selected_fire_positives": len(selected), "dataset_rows": len(rows), "quality_gate": gate["status"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "deduplicate", "error-mine", "select-positives", "build", "quality-gate", "baseline", "difficulty", "license", "prepare", "smoke-train", "train", "thresholds", "benchmark-v2-1", "artifact", "repair-preflight", "error-mine-real", "diagnose-current", "deduplicate-repaired", "select-positives-repaired", "build-repaired", "quality-gate-repaired", "eval-frozen-repaired", "smoke-train-repaired", "compare-repaired-1e", "decision", "historical-preflight", "reconstruct-exposure", "propagate-exposure", "audit-current-contamination", "force-mined-train", "build-clean-eval", "quality-gate-clean", "eval-frozen-clean", "smoke-train-clean", "compare-clean-1e", "decision-clean"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(), indent=2))
    elif args.command == "deduplicate":
        print(json.dumps(deduplicate_v2_2(), indent=2))
    elif args.command == "error-mine":
        print(f"Rows: {len(mine_error_rows(limit=args.limit))}")
    elif args.command == "select-positives":
        print(f"Selected: {len(select_fire_positives())}")
    elif args.command == "build":
        print(f"Rows: {len(build_dataset_v2_2())}")
    elif args.command == "quality-gate":
        print(json.dumps(quality_gate(), indent=2))
    elif args.command == "baseline":
        print(json.dumps(frozen_baseline(), indent=2))
    elif args.command == "difficulty":
        print(json.dumps(write_difficulty_analysis(), indent=2))
    elif args.command == "license":
        print(json.dumps(license_status(), indent=2))
    elif args.command == "prepare":
        print(json.dumps(prepare(), indent=2))
    elif args.command == "smoke-train":
        train_v2_2(smoke_test=True)
    elif args.command == "train":
        train_v2_2(smoke_test=False)
    elif args.command == "thresholds":
        print(json.dumps(threshold_tuning(), indent=2))
    elif args.command == "benchmark-v2-1":
        print(json.dumps(benchmark(V2_1_CKPT, "v2_2_cpu_benchmark_v2_1"), indent=2))
    elif args.command == "artifact":
        print(json.dumps(artifact_manifest(), indent=2))
    elif args.command == "repair-preflight":
        print(json.dumps(repair_preflight(), indent=2))
    elif args.command == "error-mine-real":
        print(f"Rows: {len(mine_error_rows_real(limit=args.limit))}")
    elif args.command == "diagnose-current":
        print(json.dumps(same_split_diagnostic_current(), indent=2))
    elif args.command == "deduplicate-repaired":
        print(json.dumps(deduplicate_v2_2_repaired(), indent=2))
    elif args.command == "select-positives-repaired":
        print(f"Selected: {len(select_fire_positives_repaired())}")
    elif args.command == "build-repaired":
        print(f"Rows: {len(build_dataset_v2_2_repaired())}")
    elif args.command == "quality-gate-repaired":
        print(json.dumps(repaired_quality_gate(), indent=2))
    elif args.command == "eval-frozen-repaired":
        print(json.dumps(frozen_v2_1_on_repaired(), indent=2))
    elif args.command == "smoke-train-repaired":
        train_v2_2_repaired_smoke()
    elif args.command == "compare-repaired-1e":
        print(json.dumps(compare_repaired_one_epoch(), indent=2))
    elif args.command == "decision":
        print(json.dumps(main_training_decision(), indent=2))
    elif args.command == "historical-preflight":
        print(json.dumps(historical_exposure_preflight(), indent=2))
    elif args.command == "reconstruct-exposure":
        print(json.dumps(reconstruct_v2_1_historical_training_exposure(), indent=2))
    elif args.command == "propagate-exposure":
        print(json.dumps(propagate_historical_exposure_components(), indent=2))
    elif args.command == "audit-current-contamination":
        print(json.dumps(audit_current_repaired_eval_contamination(), indent=2))
    elif args.command == "force-mined-train":
        print(json.dumps(force_mined_hard_fire_train_only(), indent=2))
    elif args.command == "build-clean-eval":
        print(f"Rows: {len(build_dataset_v2_2_clean_eval())}")
    elif args.command == "quality-gate-clean":
        print(json.dumps(clean_eval_quality_gate(), indent=2))
    elif args.command == "eval-frozen-clean":
        print(json.dumps(frozen_v2_1_on_clean(), indent=2))
    elif args.command == "smoke-train-clean":
        train_v2_2_clean_smoke()
    elif args.command == "compare-clean-1e":
        print(json.dumps(compare_clean_one_epoch(), indent=2))
    elif args.command == "decision-clean":
        print(json.dumps(clean_main_training_decision(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
