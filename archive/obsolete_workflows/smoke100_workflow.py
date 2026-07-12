#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
(ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / ".cache" / "ultralytics").mkdir(parents=True, exist_ok=True)

from fire_smoke_cpu.annotations import YoloBox, parse_yolo_label, write_yolo_label
from fire_smoke_cpu.provenance import assert_real_training_origins

import v2_2_workflow as v22

WORKSPACE = "smoke-detection"
PROJECT = "smoke100-uwe4t"
VERSION = 4
SOURCE_DATASET = "roboflow_smoke100_v4"
RAW_ROOT = ROOT / "data/raw/roboflow_smoke100"
CANON_ROOT = ROOT / "data/processed/canonical/roboflow_smoke100_v4"
CHALLENGER_DATASET = ROOT / "data/processed/fire_smoke_smoke100_challenger"
CHALLENGER_BUILDING = ROOT / "data/processed/fire_smoke_smoke100_challenger.building"
MANIFEST = ROOT / "data/manifests/smoke100_samples.csv"
CHALLENGER_MANIFEST = ROOT / "data/manifests/smoke100_challenger_all_samples.csv"
REPORT = ROOT / "reports"
EXPECTED_V2_1_SHA256 = v22.EXPECTED_V2_1_SHA256
V2_1_CKPT = v22.V2_1_CKPT
IMAGE_EXTS = v22.IMAGE_EXTS
SPLITS = ("train", "val", "test")
LICENSE_REVIEW = "requires_review; use Roboflow metadata/provider terms; commercial SecureVU use requires legal review"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()
    except Exception:
        return ""


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_v2_1() -> None:
    if not V2_1_CKPT.exists():
        raise SystemExit(f"ABORT: missing frozen V2.1 checkpoint: {V2_1_CKPT}")
    actual = sha256_file(V2_1_CKPT)
    if actual != EXPECTED_V2_1_SHA256:
        raise SystemExit(f"ABORT: frozen V2.1 SHA mismatch: {actual}")


def image_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_dotenv_if_needed() -> None:
    if os.environ.get("ROBOFLOW_API_KEY"):
        return
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "ROBOFLOW_API_KEY":
            os.environ[key] = value.strip().strip("'\"")


def parse_data_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    names: dict[int, str] = {}
    in_names = False
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("names:"):
            in_names = True
            rest = stripped.split(":", 1)[1].strip()
            if rest.startswith("[") and rest.endswith("]"):
                for idx, name in enumerate(rest.strip("[]").split(",")):
                    names[idx] = name.strip().strip("'\"")
            continue
        if in_names:
            m = re.match(r"(\d+)\s*:\s*(.+)", stripped)
            if m:
                names[int(m.group(1))] = m.group(2).strip().strip("'\"")
                continue
            if not raw.startswith((" ", "\t", "-")):
                in_names = False
    return {"path": str(path), "names": names, "text_sample": text[:2000]}


def find_dataset_root(raw_root: Path = RAW_ROOT) -> Path | None:
    candidates = [raw_root, raw_root / "download", raw_root / f"{PROJECT}-{VERSION}"]
    for candidate in candidates:
        if (candidate / "data.yaml").exists():
            return candidate
    for yaml in raw_root.rglob("data.yaml"):
        if "__MACOSX" not in yaml.parts:
            return yaml.parent
    return None


def detect_manual_export(raw_root: Path = RAW_ROOT) -> dict:
    root = find_dataset_root(raw_root)
    if not root:
        return {"status": "NOT_FOUND", "searched": [safe_rel(raw_root), safe_rel(raw_root / "download")]}
    yaml_meta = parse_data_yaml(root / "data.yaml")
    split_counts = {split: len(image_paths(root / split / "images")) for split in SPLITS}
    return {"status": "FOUND", "dataset_root": safe_rel(root), "image_counts_by_split": split_counts, "detected_classes": yaml_meta.get("names", {})}


def try_roboflow_cli_download(api_key: str, attempted_formats: list[str]) -> tuple[str, list[dict]]:
    cli = shutil.which("roboflow") or str(ROOT / ".venv/bin/roboflow")
    if not Path(cli).exists() and shutil.which("roboflow") is None:
        return "", [{"method": "roboflow_cli", "error": "cli_missing"}]
    errors: list[dict] = []
    for fmt in attempted_formats:
        if (RAW_ROOT / "download").exists():
            shutil.rmtree(RAW_ROOT / "download")
        cmd = [
            cli,
            "--quiet",
            "--api-key",
            api_key,
            "download",
            "-f",
            fmt,
            "-l",
            str(RAW_ROOT / "download"),
            f"{WORKSPACE}/{PROJECT}/{VERSION}",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
        stdout = (proc.stdout or "").replace(api_key, "<redacted>")[:1000]
        stderr = (proc.stderr or "").replace(api_key, "<redacted>")[:1000]
        if proc.returncode == 0 and find_dataset_root(RAW_ROOT):
            return fmt, [{"method": "roboflow_cli", "format": fmt, "returncode": proc.returncode, "stdout_sample": stdout, "stderr_sample": stderr}]
        errors.append({"method": "roboflow_cli", "format": fmt, "returncode": proc.returncode, "stdout_sample": stdout, "stderr_sample": stderr})
    return "", errors


def download_smoke100() -> dict:
    verify_v2_1()
    load_dotenv_if_needed()
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY is required; refusing to download without it")
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    attempted_formats = ["yolov11", "yolov8", "yolov5pytorch", "yolov5"]
    errors: list[dict] = []
    selected_format = ""
    sdk_log_sample = ""
    selected_format, cli_attempts = try_roboflow_cli_download(api_key, attempted_formats)
    errors.extend(cli_attempts)
    if selected_format:
        root = find_dataset_root(RAW_ROOT)
        yaml_meta = parse_data_yaml(root / "data.yaml") if root else {}
        split_counts = {split: len(image_paths(root / split / "images")) if root else 0 for split in SPLITS}
        payload = {
            "status": "PASS",
            "timestamp_utc": utc_now(),
            "source": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION, "url": f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}/dataset/{VERSION}"},
            "selected_export_format": selected_format,
            "download_method": "roboflow_cli",
            "attempted_formats": attempted_formats,
            "raw_root": safe_rel(RAW_ROOT),
            "dataset_root": safe_rel(root) if root else "",
            "image_counts_by_split": split_counts,
            "detected_classes": yaml_meta.get("names", {}),
            "metadata": {"data_yaml": yaml_meta},
            "api_key_printed": False,
            "errors": errors,
        }
        write_json(REPORT / "smoke100_download.json", payload)
        (REPORT / "smoke100_download.md").write_text("# Smoke100 Download\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload
    try:
        from roboflow import Roboflow

        for fmt in [f for f in attempted_formats if f != "yolov11"]:
            if (RAW_ROOT / "download").exists():
                shutil.rmtree(RAW_ROOT / "download")
            stream = StringIO()
            try:
                with redirect_stdout(stream), redirect_stderr(stream):
                    rf = Roboflow(api_key=api_key)
                    dataset = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(fmt, location=str(RAW_ROOT / "download"), overwrite=True)
                selected_format = fmt
                sdk_log_sample = stream.getvalue().replace(api_key, "<redacted>")[:2000]
                root = Path(getattr(dataset, "location", RAW_ROOT / "download"))
                if root.exists():
                    break
                errors.append({"format": fmt, "method": "roboflow_sdk", "error": "download_location_missing"})
                selected_format = ""
            except Exception as exc:
                errors.append({"format": fmt, "method": "roboflow_sdk", "error": type(exc).__name__, "message": str(exc).replace(api_key, "<redacted>")[:500]})
                sdk_log_sample = stream.getvalue().replace(api_key, "<redacted>")[:2000]
    except Exception as exc:
        errors.append({"method": "roboflow_sdk_import", "error": type(exc).__name__})
    if selected_format:
        root = find_dataset_root(RAW_ROOT)
        yaml_meta = parse_data_yaml(root / "data.yaml") if root else {}
        split_counts = {split: len(image_paths(root / split / "images")) if root else 0 for split in SPLITS}
        payload = {
            "status": "PASS",
            "timestamp_utc": utc_now(),
            "source": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION, "url": f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}/dataset/{VERSION}"},
            "selected_export_format": selected_format,
            "download_method": "roboflow_sdk",
            "attempted_formats": attempted_formats,
            "raw_root": safe_rel(RAW_ROOT),
            "dataset_root": safe_rel(root) if root else "",
            "image_counts_by_split": split_counts,
            "detected_classes": yaml_meta.get("names", {}),
            "metadata": {"data_yaml": yaml_meta, "sdk_log_sample_redacted": sdk_log_sample},
            "api_key_printed": False,
            "errors": errors,
        }
        write_json(REPORT / "smoke100_download.json", payload)
        (REPORT / "smoke100_download.md").write_text("# Smoke100 Download\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload

    zip_path = RAW_ROOT / "smoke100_v4_download.zip"
    for fmt in [f for f in attempted_formats if f != "yolov11"]:
        url = f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}/dataset/{VERSION}/download/{fmt}?key={api_key}"
        safe_url = url.replace(api_key, "<redacted>")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SecureVU-Smoke100-Ingest/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, zip_path.open("wb") as out:
                shutil.copyfileobj(resp, out)
            if zip_path.stat().st_size > 1024:
                selected_format = fmt
                break
            errors.append({"format": fmt, "url": safe_url, "error": "download_too_small"})
        except urllib.error.HTTPError as exc:
            errors.append({"format": fmt, "url": safe_url, "error": f"http_{exc.code}"})
        except Exception as exc:
            errors.append({"format": fmt, "url": safe_url, "error": type(exc).__name__})
    if not selected_format:
        payload = {"status": "FAIL", "attempted_formats": attempted_formats, "errors": errors, "manual_export": detect_manual_export(RAW_ROOT), "api_key_printed": False}
        write_json(REPORT / "smoke100_download.json", payload)
        raise SystemExit("Smoke100 download failed; see reports/smoke100_download.json")
    extract_root = RAW_ROOT / "download"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_root)
    root = find_dataset_root(RAW_ROOT)
    yaml_meta = parse_data_yaml(root / "data.yaml") if root else {}
    split_counts = {split: len(image_paths(root / split / "images")) if root else 0 for split in SPLITS}
    payload = {
        "status": "PASS",
        "timestamp_utc": utc_now(),
        "source": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION, "url": f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}/dataset/{VERSION}"},
        "selected_export_format": selected_format,
        "attempted_formats": attempted_formats,
        "raw_root": safe_rel(RAW_ROOT),
        "dataset_root": safe_rel(root) if root else "",
        "zip_path": safe_rel(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "image_counts_by_split": split_counts,
        "detected_classes": yaml_meta.get("names", {}),
        "metadata": {"data_yaml": yaml_meta},
        "api_key_printed": False,
        "errors": errors,
    }
    write_json(REPORT / "smoke100_download.json", payload)
    (REPORT / "smoke100_download.md").write_text("# Smoke100 Download\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def raw_split_for(image: Path, root: Path) -> str:
    try:
        rel = image.relative_to(root)
    except ValueError:
        return ""
    return rel.parts[0] if rel.parts else ""


def label_for(image: Path, root: Path) -> Path:
    split = raw_split_for(image, root)
    return root / split / "labels" / f"{image.stem}.txt"


def sequence_id(image: Path) -> tuple[str, str, str]:
    stem = image.stem.lower()
    base = stem.split(".rf.", 1)[0]
    video_match = re.match(r"(.+?(?:mp4|mov|avi|mkv))[-_ ]?\d+$", base)
    if video_match:
        video = video_match.group(1)
        return f"{SOURCE_DATASET}:video:{video}", video, video
    prefix = re.match(r"(.+?)[-_ ]\d{2,7}$", base)
    if prefix:
        scene = prefix.group(1)
        return f"{SOURCE_DATASET}:sequence:{scene}", scene, ""
    return f"{SOURCE_DATASET}:image:{base}", base, ""


def canonicalize_boxes(boxes: list[YoloBox], class_names: dict[int, str]) -> tuple[list[YoloBox], list[str]]:
    out: list[YoloBox] = []
    errors: list[str] = []
    for box in boxes:
        name = class_names.get(box.class_id, "").strip().lower()
        if name == "smoke" or (not class_names and box.class_id in {0, 1}):
            out.append(YoloBox(1, box.x_center, box.y_center, box.width, box.height))
        else:
            errors.append(f"unsupported_class:{box.class_id}:{name or '<unknown>'}")
    return out, errors


def audit_source() -> dict:
    verify_v2_1()
    root = find_dataset_root()
    if root is None:
        raise SystemExit("Smoke100 raw dataset not found; run download first")
    yaml_meta = parse_data_yaml(root / "data.yaml")
    names = {int(k): v for k, v in yaml_meta.get("names", {}).items()}
    rows = []
    corrupt = []
    missing_labels = []
    invalid_labels = []
    empty_labels = []
    class_counts = Counter()
    split_counts = Counter()
    sha_seen: dict[str, list[str]] = defaultdict(list)
    phash_seen: dict[str, list[str]] = defaultdict(list)
    for image in image_paths(root):
        if "/images/" not in image.as_posix():
            continue
        split = raw_split_for(image, root)
        split_counts[split] += 1
        width, height = v22.image_dimensions(image)
        if width <= 0 or height <= 0:
            corrupt.append(safe_rel(image))
            continue
        label = label_for(image, root)
        if not label.exists():
            missing_labels.append(safe_rel(image))
            continue
        boxes, errors = parse_yolo_label(label)
        if errors:
            invalid_labels.append({"image": safe_rel(image), "label": safe_rel(label), "errors": errors})
        if not boxes and not errors:
            empty_labels.append(safe_rel(image))
        for box in boxes:
            class_counts[str(box.class_id)] += 1
        sha = sha256_file(image)
        ph = v22.phash(image)
        sha_seen[sha].append(safe_rel(image))
        if ph:
            phash_seen[ph].append(safe_rel(image))
        rows.append({"image": image, "label": label, "split": split, "sha": sha, "phash": ph})
    duplicate_groups = [v for v in sha_seen.values() if len(v) > 1]
    near_pairs = []
    items = [(r["phash"], safe_rel(r["image"])) for r in rows if r["phash"]]
    for i, (ph_a, img_a) in enumerate(items):
        for ph_b, img_b in items[i + 1 :]:
            dist = v22.hamming_hex(ph_a, ph_b)
            if dist <= 4:
                near_pairs.append({"image_a": img_a, "image_b": img_b, "hamming_distance": dist})
                if len(near_pairs) >= 1000:
                    break
        if len(near_pairs) >= 1000:
            break
    license_files = [safe_rel(p) for p in root.rglob("*") if p.is_file() and "license" in p.name.lower()]
    suitable_rd = not corrupt and not invalid_labels and not missing_labels and bool(rows)
    payload = {
        "status": "PASS" if suitable_rd else "FAIL",
        "timestamp_utc": utc_now(),
        "dataset_root": safe_rel(root),
        "class_names": names,
        "class_ids": sorted(class_counts),
        "raw_image_counts_by_split": dict(split_counts),
        "box_counts_by_raw_class_id": dict(class_counts),
        "smoke_boxes": sum(count for cid, count in class_counts.items() if names.get(int(cid), "").lower() == "smoke" or cid == "0"),
        "corrupt_images": len(corrupt),
        "missing_labels": len(missing_labels),
        "invalid_label_files": len(invalid_labels),
        "empty_labels": len(empty_labels),
        "exact_duplicate_groups": len(duplicate_groups),
        "near_duplicate_pairs_sampled": len(near_pairs),
        "possible_video_or_sequence_groups": "inferred during prepare via filename pattern",
        "license_files": license_files,
        "license_status": LICENSE_REVIEW,
        "suitability": {
            "rd_training": "suitable" if suitable_rd else "not_suitable_until_errors_fixed",
            "commercial_securevu_training": "requires_licensing_review",
            "licensing_review_required": True,
        },
        "samples": {"corrupt": corrupt[:100], "missing_labels": missing_labels[:100], "invalid_labels": invalid_labels[:100], "duplicates": duplicate_groups[:50], "near_pairs": near_pairs[:50]},
    }
    write_json(REPORT / "smoke100_source_audit.json", payload)
    (REPORT / "smoke100_source_audit.md").write_text("# Smoke100 Source Audit\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


SMOKE100_FIELDS = [
    "sample_id", "canonical_image_path", "canonical_label_path", "source_dataset", "original_image_path", "original_label_path", "original_filename", "sha256", "perceptual_hash", "dhash", "component_id", "width", "height", "has_fire", "has_smoke", "fire_box_count", "smoke_box_count", "is_negative", "object_size_bucket", "is_synthetic", "is_cctv_like", "group_id", "scene_id", "video_id", "license_status", "provenance_status", "data_origin", "raw_split", "split", "assigned_split", "excluded", "exclusion_reason",
]


def prepare_smoke100() -> dict:
    audit = audit_source()
    if audit["status"] != "PASS":
        raise SystemExit("Refusing to prepare Smoke100: source audit did not PASS")
    root = find_dataset_root()
    assert root is not None
    names = {int(k): v for k, v in parse_data_yaml(root / "data.yaml").get("names", {}).items()}
    if names and not any(v.strip().lower() == "smoke" for v in names.values()):
        raise SystemExit(f"Refusing Smoke100 prepare: no smoke class found in {names}")
    rows: list[dict] = []
    excluded: list[dict] = []
    if CANON_ROOT.exists():
        shutil.rmtree(CANON_ROOT)
    for split in SPLITS:
        for sub in ("images", "labels"):
            (CANON_ROOT / sub).mkdir(parents=True, exist_ok=True)
    for image in image_paths(root):
        if "/images/" not in image.as_posix():
            continue
        raw_split = raw_split_for(image, root)
        label = label_for(image, root)
        if not label.exists():
            excluded.append({"image": safe_rel(image), "reason": "missing_label_not_negative"})
            continue
        boxes, errors = parse_yolo_label(label)
        if errors:
            excluded.append({"image": safe_rel(image), "label": safe_rel(label), "reason": ";".join(errors)})
            continue
        boxes, class_errors = canonicalize_boxes(boxes, names)
        if class_errors:
            excluded.append({"image": safe_rel(image), "label": safe_rel(label), "reason": ";".join(class_errors)})
            continue
        sha = sha256_file(image)
        sample_id = f"{SOURCE_DATASET}_{image.stem}_{sha[:12]}".replace("/", "_")
        image_out = CANON_ROOT / "images" / f"{sample_id}{image.suffix.lower() or '.jpg'}"
        label_out = CANON_ROOT / "labels" / f"{sample_id}.txt"
        shutil.copy2(image, image_out)
        write_yolo_label(label_out, boxes)
        width, height = v22.image_dimensions(image_out)
        group_id, scene_id, video_id = sequence_id(image)
        smoke_count = len(boxes)
        rows.append({
            "sample_id": sample_id,
            "canonical_image_path": str(image_out),
            "canonical_label_path": str(label_out),
            "source_dataset": SOURCE_DATASET,
            "original_image_path": str(image),
            "original_label_path": str(label),
            "original_filename": image.name,
            "sha256": sha256_file(image_out),
            "perceptual_hash": v22.phash(image_out),
            "dhash": v22.dhash(image_out),
            "component_id": group_id,
            "width": width,
            "height": height,
            "has_fire": False,
            "has_smoke": smoke_count > 0,
            "fire_box_count": 0,
            "smoke_box_count": smoke_count,
            "is_negative": smoke_count == 0,
            "object_size_bucket": "",
            "is_synthetic": False,
            "is_cctv_like": False,
            "group_id": group_id,
            "scene_id": scene_id,
            "video_id": video_id,
            "license_status": LICENSE_REVIEW,
            "provenance_status": "roboflow_universe_dataset_v4",
            "data_origin": "roboflow_universe",
            "raw_split": raw_split,
            "split": "",
            "assigned_split": "",
            "excluded": False,
            "exclusion_reason": "",
        })
    write_csv(MANIFEST, rows, SMOKE100_FIELDS)
    payload = {
        "status": "PASS" if rows else "FAIL",
        "timestamp_utc": utc_now(),
        "usable_samples": len(rows),
        "excluded_samples": len(excluded),
        "class_mapping": {"raw": parse_data_yaml(root / "data.yaml").get("names", {}), "canonical": {"1": "smoke"}, "fire_labels_created": 0},
        "license_status": LICENSE_REVIEW,
        "manifest": safe_rel(MANIFEST),
        "excluded_sample": excluded[:100],
    }
    write_json(REPORT / "smoke100_prepare.json", payload)
    (REPORT / "smoke100_prepare.md").write_text("# Smoke100 Prepare\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def existing_rows_for_duplicate_audit() -> list[dict]:
    rows = []
    for path in [
        ROOT / "data/manifests/all_samples.csv",
        ROOT / "data/manifests/v2_1_real_all_samples.csv",
        ROOT / "data/manifests/v2_2_all_samples.csv",
        ROOT / "data/manifests/v2_2_clean_eval_all_samples.csv",
    ]:
        for row in read_csv(path):
            row = dict(row)
            row["_manifest"] = safe_rel(path)
            rows.append(row)
    return rows


def duplicate_audit() -> dict:
    rows = read_csv(MANIFEST)
    if not rows:
        raise SystemExit("Missing Smoke100 manifest; run prepare first")
    existing = existing_rows_for_duplicate_audit()
    by_sha = defaultdict(list)
    for row in existing:
        if row.get("sha256"):
            by_sha[row["sha256"]].append(row)
    exact = []
    clean_eval_exact = []
    for row in rows:
        matches = by_sha.get(row.get("sha256", ""), [])
        if matches:
            item = {"smoke100_sample_id": row["sample_id"], "matches": [{"sample_id": m.get("sample_id"), "split": m.get("split"), "manifest": m.get("_manifest")} for m in matches]}
            exact.append(item)
            if any(m.get("_manifest", "").endswith("v2_2_clean_eval_all_samples.csv") and m.get("split") in {"val", "test"} for m in matches):
                clean_eval_exact.append(item)
    near = []
    clean_eval_near = []
    existing_phash = [(r.get("perceptual_hash"), r) for r in existing if r.get("perceptual_hash")]
    dsu = v22.DSU()
    for row in rows:
        dsu.add(row["sample_id"])
    for i, row in enumerate(rows):
        if row.get("perceptual_hash"):
            for ph, ex in existing_phash:
                dist = v22.hamming_hex(row["perceptual_hash"], ph)
                if dist <= 4:
                    item = {"smoke100_sample_id": row["sample_id"], "existing_sample_id": ex.get("sample_id"), "existing_split": ex.get("split"), "existing_manifest": ex.get("_manifest"), "hamming_distance": dist}
                    near.append(item)
                    if ex.get("_manifest", "").endswith("v2_2_clean_eval_all_samples.csv") and ex.get("split") in {"val", "test"}:
                        clean_eval_near.append(item)
        for other in rows[i + 1 :]:
            if row.get("perceptual_hash") and other.get("perceptual_hash") and v22.hamming_hex(row["perceptual_hash"], other["perceptual_hash"]) <= 4:
                dsu.union(row["sample_id"], other["sample_id"])
    comp_rows = []
    for idx, members in enumerate(dsu.groups().values(), start=1):
        cid = f"smoke100_component_{idx:06d}"
        for sample_id in members:
            comp_rows.append({"component_id": cid, "sample_id": sample_id, "component_size": len(members)})
    comp_map = {r["sample_id"]: r["component_id"] for r in comp_rows}
    for row in rows:
        row["component_id"] = comp_map.get(row["sample_id"], row.get("component_id", row["sample_id"]))
    write_csv(MANIFEST, rows, SMOKE100_FIELDS)
    write_csv(ROOT / "data/manifests/smoke100_duplicate_components.csv", comp_rows, ["component_id", "sample_id", "component_size"])
    payload = {
        "status": "PASS" if not clean_eval_exact and not clean_eval_near else "FAIL",
        "timestamp_utc": utc_now(),
        "exact_duplicates_against_existing": len(exact),
        "near_duplicates_against_existing": len(near),
        "exact_duplicates_against_clean_eval_val_test": len(clean_eval_exact),
        "near_duplicates_against_clean_eval_val_test": len(clean_eval_near),
        "smoke100_internal_components": len({r["component_id"] for r in comp_rows}),
        "leakage_detected": bool(clean_eval_exact or clean_eval_near),
        "samples": {"exact": exact[:100], "near": near[:100], "clean_eval_exact": clean_eval_exact[:100], "clean_eval_near": clean_eval_near[:100]},
    }
    write_json(REPORT / "smoke100_duplicate_audit.json", payload)
    (REPORT / "smoke100_duplicate_audit.md").write_text("# Smoke100 Duplicate Audit\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def copy_sample(row: dict, split: str, source_prefix: str) -> dict:
    image = Path(row["canonical_image_path"])
    label = Path(row["canonical_label_path"])
    sample_id = f"{source_prefix}_{row['sample_id']}"
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


def build_challenger() -> list[dict]:
    verify_v2_1()
    dup = duplicate_audit()
    if dup["status"] != "PASS":
        raise SystemExit("Refusing to build challenger: Smoke100 duplicate audit failed")
    clean_rows = read_csv(ROOT / "data/manifests/v2_2_clean_eval_all_samples.csv")
    smoke_rows = [r for r in read_csv(MANIFEST) if str(r.get("excluded", "")).lower() not in {"1", "true", "yes"}]
    if not clean_rows or not smoke_rows:
        raise SystemExit("Missing clean eval or Smoke100 rows")
    if CHALLENGER_BUILDING.exists():
        shutil.rmtree(CHALLENGER_BUILDING)
    final_rows: list[dict] = []
    for row in clean_rows:
        final_rows.append(copy_sample(row, row["split"], "clean"))
    for row in smoke_rows:
        final_rows.append(copy_sample(row, "train", "smoke100"))
    assert_real_training_origins(final_rows)
    if CHALLENGER_DATASET.exists():
        shutil.rmtree(CHALLENGER_DATASET)
    (CHALLENGER_BUILDING / "fire_smoke.yaml").write_text("\n".join([f"path: {CHALLENGER_DATASET}", "train: images/train", "val: images/val", "test: images/test", "names:", "  0: fire", "  1: smoke", ""]), encoding="utf-8")
    fields = sorted({k for row in final_rows for k in row.keys()})
    write_csv(CHALLENGER_MANIFEST, final_rows, fields)
    CHALLENGER_BUILDING.rename(CHALLENGER_DATASET)
    comp = composition(final_rows)
    write_json(REPORT / "smoke100_challenger_composition.json", comp)
    (REPORT / "smoke100_challenger_composition.md").write_text("# Smoke100 Challenger Composition\n\n" + json.dumps(comp, indent=2) + "\n", encoding="utf-8")
    return final_rows


def composition(rows: list[dict]) -> dict:
    by_source_split = defaultdict(Counter)
    split_counts = Counter()
    buckets = {split: Counter() for split in SPLITS}
    boxes = {split: Counter() for split in SPLITS}
    object_size = {split: Counter() for split in SPLITS}
    for row in rows:
        split = row.get("split", "")
        source = row.get("source_dataset", "")
        split_counts[split] += 1
        by_source_split[source][split] += 1
        bucket = v22.row_bucket(row)
        buckets[split][bucket] += 1
        boxes[split]["fire"] += int(row.get("fire_box_count") or 0)
        boxes[split]["smoke"] += int(row.get("smoke_box_count") or 0)
        object_size[split][row.get("object_size_bucket") or "none"] += 1
    return {
        "timestamp_utc": utc_now(),
        "total_images": len(rows),
        "images_by_source": {source: dict(counter) for source, counter in by_source_split.items()},
        "images_by_split": dict(split_counts),
        "bucket_counts_by_split": {split: dict(counter) for split, counter in buckets.items()},
        "boxes_by_split": {split: dict(counter) for split, counter in boxes.items()},
        "object_size_by_split": {split: dict(counter) for split, counter in object_size.items()},
        "dataset": safe_rel(CHALLENGER_DATASET),
        "manifest": safe_rel(CHALLENGER_MANIFEST),
    }


def quality_gate(tests_passed: bool = False) -> dict:
    rows = read_csv(CHALLENGER_MANIFEST)
    dup = json.loads((REPORT / "smoke100_duplicate_audit.json").read_text(encoding="utf-8")) if (REPORT / "smoke100_duplicate_audit.json").exists() else {}
    source = json.loads((REPORT / "smoke100_source_audit.json").read_text(encoding="utf-8")) if (REPORT / "smoke100_source_audit.json").exists() else {}
    clean_gate = json.loads((REPORT / "v2_2_clean_eval_quality_gate.json").read_text(encoding="utf-8")) if (REPORT / "v2_2_clean_eval_quality_gate.json").exists() else {}
    leaks = v22.leakage_counts(rows)
    gates = {
        "V2_1_CHECKPOINT_SHA_VERIFIED": V2_1_CKPT.exists() and sha256_file(V2_1_CKPT) == EXPECTED_V2_1_SHA256,
        "NO_EXACT_CROSS_SPLIT_DUPLICATE_LEAKAGE": leaks["exact_cross_split_duplicates"] == 0,
        "NO_NEAR_DUPLICATE_COMPONENT_LEAKAGE": leaks["near_duplicate_components_spanning_splits"] == 0,
        "NO_MISSING_ANNOTATION_TREATED_AS_NEGATIVE": source.get("missing_labels", 0) == 0,
        "CANONICAL_MAPPING_VERIFIED": all(str(r.get("has_fire")).lower() in {"false", "0"} and int(r.get("fire_box_count") or 0) == 0 for r in read_csv(MANIFEST)),
        "NO_MOCK_DATA": not any(str(r.get("data_origin")) == "mock" for r in rows),
        "NO_PLACEHOLDER_DATA": not any(str(r.get("data_origin")) == "placeholder" for r in rows),
        "NO_UNKNOWN_DATA_ORIGIN": not any(str(r.get("data_origin")) in {"", "unknown"} for r in rows),
        "NO_CLEAN_EVAL_CONTAMINATION_BY_V2_1_TRAINING": clean_gate.get("status") == "PASS",
        "SMOKE100_SOURCE_AUDIT_COMPLETED": source.get("status") == "PASS",
        "SMOKE100_DUPLICATE_AUDIT_PASS": dup.get("status") == "PASS",
        "SMOKE100_TRAINING_ONLY": all(r.get("split") == "train" for r in rows if r.get("source_dataset") == SOURCE_DATASET),
        "TESTS_PASS": tests_passed,
    }
    payload = {"status": "PASS" if all(gates.values()) else "FAIL", "gates": gates, "failed_gates": [k for k, v in gates.items() if not v], "dataset_rows": len(rows), **leaks}
    write_json(REPORT / "smoke100_quality_gate.json", payload)
    (REPORT / "smoke100_quality_gate.md").write_text("# Smoke100 Quality Gate\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def train_challenger() -> None:
    gate = json.loads((REPORT / "smoke100_quality_gate.json").read_text(encoding="utf-8")) if (REPORT / "smoke100_quality_gate.json").exists() else {}
    if gate.get("status") != "PASS":
        raise SystemExit(f"Refusing Smoke100 challenger training: quality gate is {gate.get('status', 'missing')}")
    from ultralytics import YOLO

    model = YOLO(str(V2_1_CKPT))
    model.train(data=str(CHALLENGER_DATASET / "fire_smoke.yaml"), epochs=1, patience=1, imgsz=512, device="cpu", batch=8, workers=2, cache=False, seed=42, optimizer="AdamW", lr0=0.0001, lrf=0.01, weight_decay=0.0005, warmup_epochs=1.0, hsv_h=0.01, hsv_s=0.30, hsv_v=0.30, translate=0.10, scale=0.30, fliplr=0.5, flipud=0.0, mosaic=0.20, close_mosaic=1, mixup=0.0, copy_paste=0.0, project="runs/detect", name="smoke100_v2_1_challenger_1e")
    ckpt = ROOT / "runs/detect/runs/detect/smoke100_v2_1_challenger_1e/weights/best.pt"
    payload = {"status": "COMPLETED", "epochs_requested": 1, "lr0": 0.0001, "run_name": "smoke100_v2_1_challenger_1e", "starting_checkpoint": safe_rel(V2_1_CKPT), "checkpoint": safe_rel(ckpt), "checkpoint_sha256": sha256_file(ckpt) if ckpt.exists() else ""}
    write_json(REPORT / "smoke100_challenger_1e_train.json", payload)
    (REPORT / "smoke100_challenger_1e_train.md").write_text("# Smoke100 Challenger 1e Train\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def compare_challenger() -> dict:
    ckpt = ROOT / "runs/detect/runs/detect/smoke100_v2_1_challenger_1e/weights/best.pt"
    baseline = v22.frozen_v2_1_on_clean()
    challenger = {
        "val": v22.evaluate_checkpoint_on_dataset(ckpt, v22.V2_2_CLEAN_DATASET, ROOT / "data/manifests/v2_2_clean_eval_all_samples.csv", "val", "smoke100_1e"),
        "test": v22.evaluate_checkpoint_on_dataset(ckpt, v22.V2_2_CLEAN_DATASET, ROOT / "data/manifests/v2_2_clean_eval_all_samples.csv", "test", "smoke100_1e"),
    }
    def delta(split: str, cls: str, metric: str) -> dict:
        before = baseline[split][cls][metric]
        after = challenger[split][cls][metric]
        absolute = after - before
        return {"absolute": absolute, "relative": absolute / before if before else None}
    payload = {
        "frozen_v2_1": baseline,
        "smoke100_challenger_1e": challenger,
        "deltas": {f"{split}_{cls}_{metric}": delta(split, cls, metric) for split in SPLITS if split != "train" for cls in ("overall", "fire", "smoke") for metric in ("precision", "recall", "mAP50", "mAP50_95")},
        "negative_fp_delta": {
            split: {
                "absolute": (challenger[split]["negatives"]["false_positive_image_rate"] or 0) - (baseline[split]["negatives"]["false_positive_image_rate"] or 0),
                "relative": (((challenger[split]["negatives"]["false_positive_image_rate"] or 0) - (baseline[split]["negatives"]["false_positive_image_rate"] or 0)) / baseline[split]["negatives"]["false_positive_image_rate"] if baseline[split]["negatives"]["false_positive_image_rate"] else None),
            }
            for split in ("val", "test")
        },
    }
    write_json(REPORT / "v2_1_vs_smoke100_challenger_1e.json", payload)
    (REPORT / "v2_1_vs_smoke100_challenger_1e.md").write_text("# V2.1 vs Smoke100 Challenger 1e\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def decision() -> dict:
    gate = json.loads((REPORT / "smoke100_quality_gate.json").read_text(encoding="utf-8")) if (REPORT / "smoke100_quality_gate.json").exists() else {}
    comp = json.loads((REPORT / "v2_1_vs_smoke100_challenger_1e.json").read_text(encoding="utf-8")) if (REPORT / "v2_1_vs_smoke100_challenger_1e.json").exists() else {}
    reasons: list[str] = []
    if gate.get("status") != "PASS":
        reasons.append("Smoke100 quality gate is not PASS")
    if not comp:
        reasons.append("comparison report missing")
    else:
        base = comp["frozen_v2_1"]["val"]
        one = comp["smoke100_challenger_1e"]["val"]
        before, after = base["smoke"]["mAP50"], one["smoke"]["mAP50"]
        if before and after < before:
            reasons.append("smoke mAP50 materially regressed")
        before, after = base["smoke"]["recall"], one["smoke"]["recall"]
        if before and after < before:
            reasons.append("smoke recall materially regressed")
        before, after = base["fire"]["recall"], one["fire"]["recall"]
        if before and (before - after) / before > 0.03:
            reasons.append("fire recall relative drop exceeds 3%")
        base_fp = base["negatives"]["false_positive_image_rate"] or 0
        one_fp = one["negatives"]["false_positive_image_rate"] or 0
        if one_fp > max(base_fp + 0.02, base_fp * 1.25):
            reasons.append("negative false-positive image rate regressed beyond limit")
        smoke_map_delta = comp["deltas"].get("val_smoke_mAP50", {}).get("relative")
        smoke_recall_delta = comp["deltas"].get("val_smoke_recall", {}).get("relative")
        if smoke_map_delta is not None and smoke_recall_delta is not None and not (smoke_map_delta > 0 and smoke_recall_delta >= 0):
            reasons.append("no meaningful smoke improvement protecting recall")
    payload = {"decision": "GO" if not reasons else "NO_GO", "failure_reasons": reasons, "long_training_started": False, "main_training_ran": False}
    write_json(REPORT / "smoke100_challenger_decision.json", payload)
    (REPORT / "smoke100_challenger_decision.md").write_text("# Smoke100 Challenger Decision\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "source-audit", "prepare", "duplicate-audit", "build-challenger", "quality-gate", "train-1e", "compare-1e", "decision"])
    parser.add_argument("--tests-passed", action="store_true")
    args = parser.parse_args()
    if args.command == "download":
        print(json.dumps(download_smoke100(), indent=2))
    elif args.command == "source-audit":
        print(json.dumps(audit_source(), indent=2))
    elif args.command == "prepare":
        print(json.dumps(prepare_smoke100(), indent=2))
    elif args.command == "duplicate-audit":
        print(json.dumps(duplicate_audit(), indent=2))
    elif args.command == "build-challenger":
        rows = build_challenger()
        print(json.dumps({"rows": len(rows)}, indent=2))
    elif args.command == "quality-gate":
        print(json.dumps(quality_gate(tests_passed=args.tests_passed), indent=2))
    elif args.command == "train-1e":
        train_challenger()
    elif args.command == "compare-1e":
        print(json.dumps(compare_challenger(), indent=2))
    elif args.command == "decision":
        print(json.dumps(decision(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
