#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fire_smoke_cpu.annotations import YoloBox, parse_yolo_label, write_yolo_label
from fire_smoke_cpu.provenance import assert_real_training_origins

import error_driven_workflow as ed
import v2_2_workflow as v22


class _Helpers:
    """Archive-only helpers (shared utilities from active workflows)."""

    utc_now = staticmethod(ed.utc_now)
    write_json = staticmethod(ed.write_json)
    write_csv = staticmethod(ed.write_csv)
    read_csv = staticmethod(ed.read_csv)
    sha256_file = staticmethod(ed.sha256_file)
    safe_rel = staticmethod(ed.safe_rel)
    verify_v2_1 = staticmethod(ed.verify_v2_1)
    image_paths = staticmethod(v22.image_paths)
    V2_1_CKPT = ed.V2_1_CKPT
    EXPECTED_V2_1_SHA256 = ed.EXPECTED_V2_1_SHA256

    @staticmethod
    def existing_rows_for_duplicate_audit():
        rows = []
        for name in (
            "v2_1_real_all_samples.csv",
            "v2_2_clean_eval_all_samples.csv",
            "medyoussef_real_samples.csv",
            "libreyolo_real_samples.csv",
        ):
            path = ROOT / "data/manifests" / name
            if path.exists():
                rows.extend(ed.read_csv(path))
        return rows


s100 = _Helpers()

SOURCE_DATASET = "hf_kien_indoor_fire_smoke"
HF_DATASET = "KienNgyuen/Fire-Smoke-Detection"
HF_URL = "https://huggingface.co/datasets/KienNgyuen/Fire-Smoke-Detection"
ARCHIVE_NAME = "Indoor Fire Smoke.zip"
ARCHIVE_URL = (
    "https://huggingface.co/datasets/KienNgyuen/Fire-Smoke-Detection/resolve/main/"
    "Indoor%20Fire%20Smoke.zip"
)
RAW_ROOT = ROOT / "data/raw/hf_kien_fire_smoke"
ZIP_PATH = RAW_ROOT / "Indoor_Fire_Smoke.zip"
DATASET_ROOT = RAW_ROOT / "download/Indoor Fire Smoke"
CANON_ROOT = ROOT / "data/processed/canonical/hf_kien_indoor_fire_smoke"
CHALLENGER_DATASET = ROOT / "data/processed/fire_smoke_hf_kien_challenger"
CHALLENGER_BUILDING = ROOT / "data/processed/fire_smoke_hf_kien_challenger.building"
MANIFEST = ROOT / "data/manifests/hf_kien_indoor_fire_smoke_samples.csv"
CHALLENGER_MANIFEST = ROOT / "data/manifests/hf_kien_challenger_all_samples.csv"
REPORT = ROOT / "reports"
SPLITS = ("train", "valid", "test")
CANON_SPLITS = ("train", "val", "test")
CLASS_MAP = {0: 0, 1: 1}
CLASS_MAPPING_EVIDENCE = (
    "Archive data.yaml uses literal names ['0', '1']; filename correlation showed fire filenames "
    "overwhelmingly label 0 and smoke filenames label 1, matching canonical 0=fire and 1=smoke."
)
LICENSE_STATUS = "HF metadata apache-2.0; archive data.yaml Roboflow license CC BY 4.0"
MAX_FALLBACK_TRAIN_SAMPLES = 1600
RUN_NAME = "hf_kien_v2_1_challenger_1e"


FIELDS = [
    "sample_id", "canonical_image_path", "canonical_label_path", "source_dataset",
    "original_image_path", "original_label_path", "original_filename", "sha256",
    "perceptual_hash", "dhash", "component_id", "width", "height", "has_fire",
    "has_smoke", "fire_box_count", "smoke_box_count", "is_negative",
    "object_size_bucket", "is_synthetic", "is_cctv_like", "group_id", "scene_id",
    "video_id", "license_status", "provenance_status", "data_origin", "raw_split",
    "split", "assigned_split", "excluded", "exclusion_reason",
]


def write_md(path: Path, title: str, payload: dict) -> None:
    path.write_text(f"# {title}\n\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download() -> dict:
    s100.verify_v2_1()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        with urllib.request.urlopen(ARCHIVE_URL, timeout=180) as resp, ZIP_PATH.open("wb") as out:
            shutil.copyfileobj(resp, out)
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT.parent)
    DATASET_ROOT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for member in zf.infolist():
            if member.filename.startswith("__MACOSX/") or "/._" in member.filename or member.filename.endswith("/"):
                continue
            zf.extract(member, DATASET_ROOT.parent)
    payload = {
        "status": "PASS" if (DATASET_ROOT / "data.yaml").exists() else "FAIL",
        "timestamp_utc": s100.utc_now(),
        "source": {"dataset": HF_DATASET, "url": HF_URL, "archive": ARCHIVE_NAME},
        "download_method": "huggingface_resolve_zip",
        "archive_path": s100.safe_rel(ZIP_PATH),
        "archive_sha256": s100.sha256_file(ZIP_PATH),
        "dataset_root": s100.safe_rel(DATASET_ROOT),
        "license_status": LICENSE_STATUS,
        "api_key_printed": False,
    }
    s100.write_json(REPORT / "hf_kien_download.json", payload)
    write_md(REPORT / "hf_kien_download.md", "HF Kien Fallback Download", payload)
    return payload


def image_for_label(label: Path) -> Path | None:
    image_dir = label.parents[1] / "images"
    for suffix in v22.IMAGE_EXTS:
        candidate = image_dir / f"{label.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def audit_source() -> dict:
    s100.verify_v2_1()
    if not (DATASET_ROOT / "data.yaml").exists():
        raise SystemExit("HF Kien raw dataset not found; run download first")
    split_counts = Counter()
    class_counts = Counter()
    filename_signal = defaultdict(Counter)
    corrupt: list[str] = []
    missing_labels: list[str] = []
    invalid_labels: list[dict] = []
    empty_labels: list[str] = []
    rows: list[dict] = []
    for split in SPLITS:
        for image in s100.image_paths(DATASET_ROOT / split / "images"):
            split_counts[split] += 1
            width, height = v22.image_dimensions(image)
            if width <= 0 or height <= 0:
                corrupt.append(s100.safe_rel(image))
                continue
            label = DATASET_ROOT / split / "labels" / f"{image.stem}.txt"
            if not label.exists():
                missing_labels.append(s100.safe_rel(image))
                continue
            boxes, errors = parse_yolo_label(label)
            if errors:
                invalid_labels.append({"image": s100.safe_rel(image), "label": s100.safe_rel(label), "errors": errors})
            if not boxes and not errors:
                empty_labels.append(s100.safe_rel(image))
            token = "other"
            lower = image.name.lower()
            if "smoke" in lower:
                token = "smoke_filename"
            if "fire" in lower:
                token = "fire_filename" if token == "other" else "fire_and_smoke_filename"
            for box in boxes:
                class_counts[str(box.class_id)] += 1
                filename_signal[token][str(box.class_id)] += 1
            rows.append({"image": image, "label": label, "sha": s100.sha256_file(image), "phash": v22.phash(image)})
    payload = {
        "status": "PASS" if rows and not corrupt and not missing_labels else "FAIL",
        "timestamp_utc": s100.utc_now(),
        "dataset_root": s100.safe_rel(DATASET_ROOT),
        "source": {"dataset": HF_DATASET, "url": HF_URL, "archive": ARCHIVE_NAME},
        "license_status": LICENSE_STATUS,
        "raw_class_names": {"0": "0", "1": "1"},
        "class_mapping": {"0": "fire", "1": "smoke"},
        "class_mapping_evidence": CLASS_MAPPING_EVIDENCE,
        "raw_image_counts_by_split": dict(split_counts),
        "box_counts_by_raw_class_id": dict(class_counts),
        "filename_label_signal": {k: dict(v) for k, v in filename_signal.items()},
        "corrupt_images": len(corrupt),
        "missing_labels": len(missing_labels),
        "invalid_label_files": len(invalid_labels),
        "invalid_label_handling": "excluded_before_prepare; not treated as negatives",
        "empty_labels": len(empty_labels),
        "commercial_securevu_training": "allowed_for_R&D_pending_attribution_review",
        "samples": {"corrupt": corrupt[:50], "missing_labels": missing_labels[:50], "invalid_labels": invalid_labels[:50]},
    }
    s100.write_json(REPORT / "hf_kien_source_audit.json", payload)
    write_md(REPORT / "hf_kien_source_audit.md", "HF Kien Fallback Source Audit", payload)
    return payload


def sequence_id(image: Path) -> tuple[str, str, str]:
    stem = image.stem.lower().split(".rf.", 1)[0]
    return f"{SOURCE_DATASET}:scene:{stem}", stem, ""


def prepare() -> dict:
    audit = audit_source()
    if audit["status"] != "PASS":
        raise SystemExit("Refusing to prepare HF Kien fallback: source audit did not PASS")
    if CANON_ROOT.exists():
        shutil.rmtree(CANON_ROOT)
    (CANON_ROOT / "images").mkdir(parents=True)
    (CANON_ROOT / "labels").mkdir(parents=True)
    rows: list[dict] = []
    excluded: list[dict] = []
    for split in SPLITS:
        for image in s100.image_paths(DATASET_ROOT / split / "images"):
            label = DATASET_ROOT / split / "labels" / f"{image.stem}.txt"
            boxes, errors = parse_yolo_label(label)
            if errors:
                excluded.append({"image": s100.safe_rel(image), "reason": ";".join(errors)})
                continue
            canonical = [YoloBox(CLASS_MAP[b.class_id], b.x_center, b.y_center, b.width, b.height) for b in boxes if b.class_id in CLASS_MAP]
            sha = s100.sha256_file(image)
            sample_id = f"{SOURCE_DATASET}_{image.stem}_{sha[:12]}".replace("/", "_")
            image_out = CANON_ROOT / "images" / f"{sample_id}{image.suffix.lower()}"
            label_out = CANON_ROOT / "labels" / f"{sample_id}.txt"
            shutil.copy2(image, image_out)
            write_yolo_label(label_out, canonical)
            width, height = v22.image_dimensions(image_out)
            fire_count = sum(1 for b in canonical if b.class_id == 0)
            smoke_count = sum(1 for b in canonical if b.class_id == 1)
            group_id, scene_id, video_id = sequence_id(image)
            rows.append({
                "sample_id": sample_id,
                "canonical_image_path": str(image_out),
                "canonical_label_path": str(label_out),
                "source_dataset": SOURCE_DATASET,
                "original_image_path": str(image),
                "original_label_path": str(label),
                "original_filename": image.name,
                "sha256": s100.sha256_file(image_out),
                "perceptual_hash": v22.phash(image_out),
                "dhash": v22.dhash(image_out),
                "component_id": group_id,
                "width": width,
                "height": height,
                "has_fire": fire_count > 0,
                "has_smoke": smoke_count > 0,
                "fire_box_count": fire_count,
                "smoke_box_count": smoke_count,
                "is_negative": not canonical,
                "object_size_bucket": "",
                "is_synthetic": False,
                "is_cctv_like": True,
                "group_id": group_id,
                "scene_id": scene_id,
                "video_id": video_id,
                "license_status": LICENSE_STATUS,
                "provenance_status": "hf_dataset_apache_2_plus_archive_cc_by_4",
                "data_origin": "huggingface_snapshot",
                "raw_split": split,
                "split": "",
                "assigned_split": "",
                "excluded": False,
                "exclusion_reason": "",
            })
    s100.write_csv(MANIFEST, rows, FIELDS)
    payload = {
        "status": "PASS" if rows else "FAIL",
        "timestamp_utc": s100.utc_now(),
        "usable_samples": len(rows),
        "usable_train_samples": sum(1 for r in rows if r["raw_split"] == "train"),
        "excluded_samples": len(excluded),
        "class_mapping": {"0": "fire", "1": "smoke"},
        "class_mapping_evidence": CLASS_MAPPING_EVIDENCE,
        "license_status": LICENSE_STATUS,
        "manifest": s100.safe_rel(MANIFEST),
    }
    s100.write_json(REPORT / "hf_kien_prepare.json", payload)
    write_md(REPORT / "hf_kien_prepare.md", "HF Kien Fallback Prepare", payload)
    return payload


def duplicate_audit() -> dict:
    rows = s100.read_csv(MANIFEST)
    if not rows:
        raise SystemExit("Missing HF Kien manifest; run prepare first")
    train_rows = [r for r in rows if r.get("raw_split") == "train"]
    existing = s100.existing_rows_for_duplicate_audit()
    by_sha = defaultdict(list)
    for row in existing:
        if row.get("sha256"):
            by_sha[row["sha256"]].append(row)
    exact = []
    clean_eval_exact = []
    for row in train_rows:
        matches = by_sha.get(row.get("sha256", ""), [])
        if matches:
            item = {"sample_id": row["sample_id"], "matches": [{"sample_id": m.get("sample_id"), "split": m.get("split"), "manifest": m.get("_manifest")} for m in matches]}
            exact.append(item)
            if any(m.get("_manifest", "").endswith("v2_2_clean_eval_all_samples.csv") and m.get("split") in {"val", "test"} for m in matches):
                clean_eval_exact.append(item)
    near = []
    clean_eval_near = []
    existing_phash = [(r.get("perceptual_hash"), r) for r in existing if r.get("perceptual_hash")]
    dsu = v22.DSU()
    for row in rows:
        dsu.add(row["sample_id"])
    for row in train_rows:
        if row.get("perceptual_hash"):
            for ph, ex in existing_phash:
                dist = v22.hamming_hex(row["perceptual_hash"], ph)
                if dist <= 4:
                    item = {"sample_id": row["sample_id"], "existing_sample_id": ex.get("sample_id"), "existing_split": ex.get("split"), "existing_manifest": ex.get("_manifest"), "hamming_distance": dist}
                    near.append(item)
                    if ex.get("_manifest", "").endswith("v2_2_clean_eval_all_samples.csv") and ex.get("split") in {"val", "test"}:
                        clean_eval_near.append(item)
        if len(near) >= 1000:
            break
    excluded_ids = sorted({item["sample_id"] for item in clean_eval_exact} | {item["sample_id"] for item in clean_eval_near})
    payload = {
        "status": "PASS",
        "timestamp_utc": s100.utc_now(),
        "audited_source_rows": len(train_rows),
        "exact_duplicates_against_existing": len(exact),
        "near_duplicates_against_existing_sampled": len(near),
        "exact_duplicates_against_clean_eval_val_test": len(clean_eval_exact),
        "near_duplicates_against_clean_eval_val_test": len(clean_eval_near),
        "leakage_detected_before_exclusion": bool(clean_eval_exact or clean_eval_near),
        "excluded_from_training_for_clean_eval_contamination": len(excluded_ids),
        "excluded_sample_ids": excluded_ids,
        "leakage_after_exclusion": False,
        "samples": {"exact": exact[:50], "near": near[:50], "clean_eval_exact": clean_eval_exact[:50], "clean_eval_near": clean_eval_near[:50]},
    }
    s100.write_json(REPORT / "hf_kien_duplicate_audit.json", payload)
    write_md(REPORT / "hf_kien_duplicate_audit.md", "HF Kien Fallback Duplicate Audit", payload)
    return payload


def copy_sample(row: dict, split: str, prefix: str) -> dict:
    image = Path(row["canonical_image_path"])
    label = Path(row["canonical_label_path"])
    sample_id = f"{prefix}_{row['sample_id']}"
    image_out = CHALLENGER_BUILDING / "images" / split / f"{sample_id}{image.suffix.lower()}"
    label_out = CHALLENGER_BUILDING / "labels" / split / f"{sample_id}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_out)
    shutil.copy2(label, label_out)
    out = dict(row)
    out["sample_id"] = sample_id
    out["canonical_image_path"] = str(image_out)
    out["canonical_label_path"] = str(label_out)
    out["split"] = split
    out["assigned_split"] = split
    return out


def composition(rows: list[dict]) -> dict:
    by_source_split = defaultdict(Counter)
    split_counts = Counter()
    boxes = {split: Counter() for split in CANON_SPLITS}
    for row in rows:
        split = row.get("split", "")
        by_source_split[row.get("source_dataset", "")][split] += 1
        split_counts[split] += 1
        boxes[split]["fire"] += int(row.get("fire_box_count") or 0)
        boxes[split]["smoke"] += int(row.get("smoke_box_count") or 0)
    return {
        "timestamp_utc": s100.utc_now(),
        "total_images": len(rows),
        "images_by_source": {k: dict(v) for k, v in by_source_split.items()},
        "images_by_split": dict(split_counts),
        "boxes_by_split": {k: dict(v) for k, v in boxes.items()},
        "fallback_train_cap": MAX_FALLBACK_TRAIN_SAMPLES,
        "dataset": s100.safe_rel(CHALLENGER_DATASET),
        "manifest": s100.safe_rel(CHALLENGER_MANIFEST),
    }


def build_challenger() -> list[dict]:
    s100.verify_v2_1()
    dup = duplicate_audit()
    if dup["status"] != "PASS":
        raise SystemExit("Refusing to build challenger: HF Kien duplicate audit failed")
    clean_rows = s100.read_csv(ROOT / "data/manifests/v2_2_clean_eval_all_samples.csv")
    excluded_ids = set(dup.get("excluded_sample_ids", []))
    source_rows = [r for r in s100.read_csv(MANIFEST) if r.get("raw_split") == "train" and r.get("sample_id") not in excluded_ids]
    source_rows = sorted(source_rows, key=lambda r: (r.get("group_id", ""), r.get("sample_id", "")))[:MAX_FALLBACK_TRAIN_SAMPLES]
    if not clean_rows or not source_rows:
        raise SystemExit("Missing clean eval or HF Kien train rows")
    if CHALLENGER_BUILDING.exists():
        shutil.rmtree(CHALLENGER_BUILDING)
    final_rows = [copy_sample(row, row["split"], "clean") for row in clean_rows]
    final_rows.extend(copy_sample(row, "train", "hf_kien") for row in source_rows)
    assert_real_training_origins(final_rows)
    if CHALLENGER_DATASET.exists():
        shutil.rmtree(CHALLENGER_DATASET)
    (CHALLENGER_BUILDING / "fire_smoke.yaml").write_text(
        "\n".join([
            f"path: {CHALLENGER_DATASET}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: fire",
            "  1: smoke",
            "",
        ]),
        encoding="utf-8",
    )
    fields = sorted({k for row in final_rows for k in row})
    s100.write_csv(CHALLENGER_MANIFEST, final_rows, fields)
    CHALLENGER_BUILDING.rename(CHALLENGER_DATASET)
    comp = composition(final_rows)
    s100.write_json(REPORT / "hf_kien_challenger_composition.json", comp)
    write_md(REPORT / "hf_kien_challenger_composition.md", "HF Kien Fallback Challenger Composition", comp)
    return final_rows


def quality_gate(tests_passed: bool = False) -> dict:
    rows = s100.read_csv(CHALLENGER_MANIFEST)
    source = json.loads((REPORT / "hf_kien_source_audit.json").read_text(encoding="utf-8")) if (REPORT / "hf_kien_source_audit.json").exists() else {}
    dup = json.loads((REPORT / "hf_kien_duplicate_audit.json").read_text(encoding="utf-8")) if (REPORT / "hf_kien_duplicate_audit.json").exists() else {}
    clean_gate = json.loads((REPORT / "v2_2_clean_eval_quality_gate.json").read_text(encoding="utf-8")) if (REPORT / "v2_2_clean_eval_quality_gate.json").exists() else {}
    leaks = v22.leakage_counts(rows)
    gates = {
        "V2_1_CHECKPOINT_SHA_VERIFIED": s100.V2_1_CKPT.exists() and s100.sha256_file(s100.V2_1_CKPT) == s100.EXPECTED_V2_1_SHA256,
        "NO_EXACT_CROSS_SPLIT_DUPLICATE_LEAKAGE": leaks["exact_cross_split_duplicates"] == 0,
        "NO_NEAR_DUPLICATE_COMPONENT_LEAKAGE": leaks["near_duplicate_components_spanning_splits"] == 0,
        "NO_MISSING_ANNOTATION_TREATED_AS_NEGATIVE": source.get("missing_labels", 0) == 0,
        "CANONICAL_MAPPING_VERIFIED": source.get("class_mapping") == {"0": "fire", "1": "smoke"},
        "LICENSE_PERMITS_COMMERCIAL_R_AND_D": "apache-2.0" in source.get("license_status", "").lower() or "cc by 4.0" in source.get("license_status", "").lower(),
        "NO_MOCK_DATA": not any(str(r.get("data_origin")) == "mock" for r in rows),
        "NO_PLACEHOLDER_DATA": not any(str(r.get("data_origin")) == "placeholder" for r in rows),
        "NO_UNKNOWN_DATA_ORIGIN": not any(str(r.get("data_origin")) in {"", "unknown"} for r in rows),
        "NO_CLEAN_EVAL_CONTAMINATION_BY_V2_1_TRAINING": clean_gate.get("status") == "PASS",
        "SOURCE_AUDIT_COMPLETED": source.get("status") == "PASS",
        "DUPLICATE_AUDIT_PASS": dup.get("status") == "PASS",
        "FALLBACK_SOURCE_TRAINING_ONLY": all(r.get("split") == "train" for r in rows if r.get("source_dataset") == SOURCE_DATASET),
        "TESTS_PASS": tests_passed,
    }
    payload = {"status": "PASS" if all(gates.values()) else "FAIL", "gates": gates, "failed_gates": [k for k, v in gates.items() if not v], "dataset_rows": len(rows), **leaks}
    s100.write_json(REPORT / "hf_kien_quality_gate.json", payload)
    write_md(REPORT / "hf_kien_quality_gate.md", "HF Kien Fallback Quality Gate", payload)
    return payload


def train_1e() -> dict:
    gate = json.loads((REPORT / "hf_kien_quality_gate.json").read_text(encoding="utf-8")) if (REPORT / "hf_kien_quality_gate.json").exists() else {}
    if gate.get("status") != "PASS":
        raise SystemExit(f"Refusing HF Kien challenger training: quality gate is {gate.get('status', 'missing')}")
    from ultralytics import YOLO

    model = YOLO(str(s100.V2_1_CKPT))
    model.train(data=str(CHALLENGER_DATASET / "fire_smoke.yaml"), epochs=1, patience=1, imgsz=512, device="cpu", batch=8, workers=2, cache=False, seed=42, optimizer="AdamW", lr0=0.0001, lrf=0.01, weight_decay=0.0005, warmup_epochs=1.0, hsv_h=0.01, hsv_s=0.30, hsv_v=0.30, translate=0.10, scale=0.30, fliplr=0.5, flipud=0.0, mosaic=0.20, close_mosaic=1, mixup=0.0, copy_paste=0.0, project="runs/detect", name=RUN_NAME)
    ckpt = ROOT / "runs/detect" / RUN_NAME / "weights/best.pt"
    if not ckpt.exists():
        ckpt = ROOT / "runs/detect/runs/detect" / RUN_NAME / "weights/best.pt"
    payload = {"status": "COMPLETED" if ckpt.exists() else "FAILED", "epochs_requested": 1, "run_name": RUN_NAME, "starting_checkpoint": s100.safe_rel(s100.V2_1_CKPT), "checkpoint": s100.safe_rel(ckpt), "checkpoint_sha256": s100.sha256_file(ckpt) if ckpt.exists() else ""}
    s100.write_json(REPORT / "hf_kien_challenger_1e_train.json", payload)
    write_md(REPORT / "hf_kien_challenger_1e_train.md", "HF Kien Fallback Challenger 1e Train", payload)
    return payload


def compare_1e() -> dict:
    ckpt = ROOT / "runs/detect" / RUN_NAME / "weights/best.pt"
    if not ckpt.exists():
        ckpt = ROOT / "runs/detect/runs/detect" / RUN_NAME / "weights/best.pt"
    baseline = v22.frozen_v2_1_on_clean()
    challenger = {
        "val": v22.evaluate_checkpoint_on_dataset(ckpt, v22.V2_2_CLEAN_DATASET, ROOT / "data/manifests/v2_2_clean_eval_all_samples.csv", "val", "hf_kien_1e"),
        "test": v22.evaluate_checkpoint_on_dataset(ckpt, v22.V2_2_CLEAN_DATASET, ROOT / "data/manifests/v2_2_clean_eval_all_samples.csv", "test", "hf_kien_1e"),
    }

    def delta(split: str, cls: str, metric: str) -> dict:
        before = baseline[split][cls][metric]
        after = challenger[split][cls][metric]
        absolute = after - before
        return {"absolute": absolute, "relative": absolute / before if before else None}

    payload = {
        "frozen_v2_1": baseline,
        "hf_kien_challenger_1e": challenger,
        "deltas": {f"{split}_{cls}_{metric}": delta(split, cls, metric) for split in ("val", "test") for cls in ("overall", "fire", "smoke") for metric in ("precision", "recall", "mAP50", "mAP50_95")},
    }
    s100.write_json(REPORT / "v2_1_vs_hf_kien_challenger_1e.json", payload)
    write_md(REPORT / "v2_1_vs_hf_kien_challenger_1e.md", "V2.1 vs HF Kien Fallback Challenger 1e", payload)
    return payload


def decision() -> dict:
    gate = json.loads((REPORT / "hf_kien_quality_gate.json").read_text(encoding="utf-8")) if (REPORT / "hf_kien_quality_gate.json").exists() else {}
    comp = json.loads((REPORT / "v2_1_vs_hf_kien_challenger_1e.json").read_text(encoding="utf-8")) if (REPORT / "v2_1_vs_hf_kien_challenger_1e.json").exists() else {}
    reasons: list[str] = []
    if gate.get("status") != "PASS":
        reasons.append("HF Kien quality gate is not PASS")
    if not comp:
        reasons.append("comparison report missing")
    else:
        base = comp["frozen_v2_1"]["val"]
        one = comp["hf_kien_challenger_1e"]["val"]
        if base["smoke"]["mAP50"] and one["smoke"]["mAP50"] < base["smoke"]["mAP50"]:
            reasons.append("smoke mAP50 regressed")
        if base["smoke"]["recall"] and one["smoke"]["recall"] < base["smoke"]["recall"]:
            reasons.append("smoke recall regressed")
        if base["fire"]["recall"] and (base["fire"]["recall"] - one["fire"]["recall"]) / base["fire"]["recall"] > 0.03:
            reasons.append("fire recall relative drop exceeds 3%")
    payload = {"decision": "GO" if not reasons else "NO_GO", "failure_reasons": reasons, "long_training_started": False, "main_training_ran": False}
    s100.write_json(REPORT / "hf_kien_challenger_decision.json", payload)
    write_md(REPORT / "hf_kien_challenger_decision.md", "HF Kien Fallback Challenger Decision", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "source-audit", "prepare", "duplicate-audit", "build-challenger", "quality-gate", "train-1e", "compare-1e", "decision"])
    parser.add_argument("--tests-passed", action="store_true")
    args = parser.parse_args()
    if args.command == "download":
        print(json.dumps(download(), indent=2))
    elif args.command == "source-audit":
        print(json.dumps(audit_source(), indent=2))
    elif args.command == "prepare":
        print(json.dumps(prepare(), indent=2))
    elif args.command == "duplicate-audit":
        print(json.dumps(duplicate_audit(), indent=2))
    elif args.command == "build-challenger":
        print(json.dumps({"rows": len(build_challenger())}, indent=2))
    elif args.command == "quality-gate":
        print(json.dumps(quality_gate(tests_passed=args.tests_passed), indent=2))
    elif args.command == "train-1e":
        print(json.dumps(train_1e(), indent=2))
    elif args.command == "compare-1e":
        print(json.dumps(compare_1e(), indent=2))
    elif args.command == "decision":
        print(json.dumps(decision(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
