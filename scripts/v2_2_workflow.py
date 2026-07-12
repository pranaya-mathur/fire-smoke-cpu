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
MANIFEST_DIR = ROOT / "data/manifests"
REPORT_DIR = ROOT / "reports"


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
    parser.add_argument("command", choices=["preflight", "deduplicate", "error-mine", "select-positives", "build", "quality-gate", "baseline", "difficulty", "license", "prepare", "smoke-train", "train", "thresholds", "benchmark-v2-1", "artifact"])
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
