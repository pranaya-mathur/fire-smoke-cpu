#!/usr/bin/env python3
"""Download / verify SecureVU priority video+image sources (no training).

Priority order:
  1. FIRESENSE fire + smoke ZIPs (Zenodo, automated)
  2. D-Fire images / labels / presplit (official README Kaggle mirror)

Never deletes or rewrites existing raw archives. Resume-safe curl downloads.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_smoke_cpu.constants import MANIFEST_DIR, RAW_DIR, REPORT_DIR
from fire_smoke_cpu.utils import VIDEO_EXTS, ensure_dirs, list_images, list_videos, sha256_file, utc_now_iso

FIRESENSE_DIR = RAW_DIR / "firesense"
DFIRE_DIR = RAW_DIR / "dfire"

FIRESENSE_ARCHIVES = {
    "fire_videos.1406.zip": {
        "url": "https://zenodo.org/records/836749/files/fire_videos.1406.zip?download=1",
        "expected_mb": 623,
        "role": "fire_videos",
        "notes": "FIRESENSE flame-detection videos: 11 positive + 16 negative (Zenodo listing).",
    },
    "smoke_videos.1407.zip": {
        "url": "https://zenodo.org/records/836749/files/smoke_videos.1407.zip?download=1",
        "expected_mb": 198,
        "role": "smoke_videos",
        "notes": "FIRESENSE smoke videos: 13 positive + 9 negative (Zenodo listing).",
    },
}

DFIRE_MANUAL = {
    "dfire_images_labels": {
        "url": "https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo",
        "dest_hint": str(DFIRE_DIR / "kaggle_smoke_fire_detection_yolo"),
        "expected": "Official D-Fire README Kaggle mirror (images + YOLO labels + splits)",
        "priority": 1,
        "od_annotations": True,
        "role": "detector_training",
    },
    "dfire_presplit": {
        "url": "https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo",
        "dest_hint": str(DFIRE_DIR / "presplit"),
        "expected": "Train/val/test splits from the same official Kaggle D-Fire mirror",
        "priority": 2,
        "od_annotations": True,
        "role": "detector_training_presplit",
    },
}

DOWNLOAD_FIELDS = [
    "dataset",
    "status",
    "source_url",
    "destination",
    "timestamp_utc",
    "archive_or_repo",
    "sha256",
    "size_bytes",
    "extracted_size_bytes",
    "message",
]
LICENSE_FIELDS = [
    "name",
    "source_url",
    "license_discovered",
    "license_file_url",
    "commercial_use_status",
    "notes",
]


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    ensure_dirs(path.parent)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: dict) -> None:
    ensure_dirs(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, payload: dict) -> None:
    ensure_dirs(path.parent)
    path.write_text(
        f"# {title}\n\n```json\n{json.dumps(payload, indent=2, sort_keys=True)}\n```\n",
        encoding="utf-8",
    )


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def curl_download(url: str, dest: Path) -> dict:
    ensure_dirs(dest.parent)
    curl = shutil.which("curl")
    if not curl:
        return {"ok": False, "message": "curl_not_found"}
    # Resume-safe; follow redirects; fail on HTTP errors.
    cmd = [
        curl,
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "-C",
        "-",
        "-o",
        str(dest),
        url,
    ]
    print(f"Downloading {url} -> {dest}")
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        return {
            "ok": False,
            "message": (result.stderr or result.stdout or "curl_failed").strip()[:2000],
            "returncode": result.returncode,
        }
    return {"ok": True, "message": "downloaded", "size_bytes": dest.stat().st_size if dest.exists() else 0}


def extract_zip(archive: Path, out_dir: Path) -> dict:
    if out_dir.exists() and any(out_dir.iterdir()):
        return {"ok": True, "status": "EXISTS", "message": "extraction already present", "out_dir": str(out_dir)}
    ensure_dirs(out_dir)
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
    except zipfile.BadZipFile as exc:
        return {"ok": False, "status": "BAD_ZIP", "message": repr(exc), "out_dir": str(out_dir)}
    return {
        "ok": True,
        "status": "EXTRACTED",
        "message": "extracted",
        "out_dir": str(out_dir),
        "extracted_size_bytes": dir_size(out_dir),
    }


def record_license_firesense() -> None:
    append_csv(
        MANIFEST_DIR / "licenses.csv",
        {
            "name": "firesense",
            "source_url": "https://zenodo.org/records/836749",
            "license_discovered": "requires_review",
            "license_file_url": "https://zenodo.org/records/836749",
            "commercial_use_status": "requires review; confirm Zenodo record license before commercial deploy",
            "notes": "FIRESENSE flame/smoke videos; video-only until frame labels exist.",
        },
        LICENSE_FIELDS,
    )


def record_license_dfire() -> None:
    append_csv(
        MANIFEST_DIR / "licenses.csv",
        {
            "name": "dfire_official",
            "source_url": "https://github.com/gaia-solutions-on-demand/DFireDataset",
            "license_discovered": "CC0-1.0",
            "license_file_url": "https://github.com/gaia-solutions-on-demand/DFireDataset/blob/main/LICENSE",
            "commercial_use_status": "confirmed by license text; still review in commercial release process",
            "notes": "Official D-Fire images/labels via README Kaggle mirror.",
        },
        LICENSE_FIELDS,
    )


def download_firesense(*, force: bool = False) -> dict:
    ensure_dirs(FIRESENSE_DIR, FIRESENSE_DIR / "extracted")
    record_license_firesense()
    results = []
    for filename, meta in FIRESENSE_ARCHIVES.items():
        dest = FIRESENSE_DIR / filename
        extract_dir = FIRESENSE_DIR / "extracted" / Path(filename).stem
        status = "SKIPPED"
        message = ""
        sha = ""
        size = 0
        extracted_size = 0
        if dest.exists() and dest.stat().st_size > 1_000_000 and not force:
            status = "EXISTS"
            message = "archive already present; not re-downloaded"
            size = dest.stat().st_size
            sha = sha256_file(dest)
        else:
            dl = curl_download(meta["url"], dest)
            if not dl["ok"]:
                status = "FAILED"
                message = dl["message"]
                results.append(
                    {
                        "dataset": f"firesense_{meta['role']}",
                        "status": status,
                        "url": meta["url"],
                        "destination": str(dest),
                        "message": message,
                    }
                )
                append_csv(
                    MANIFEST_DIR / "downloads.csv",
                    {
                        "dataset": f"firesense_{meta['role']}",
                        "status": status,
                        "source_url": meta["url"],
                        "destination": str(dest),
                        "timestamp_utc": utc_now_iso(),
                        "archive_or_repo": filename,
                        "message": message,
                    },
                    DOWNLOAD_FIELDS,
                )
                continue
            status = "DOWNLOADED"
            message = dl["message"]
            size = dest.stat().st_size
            sha = sha256_file(dest)

        ex = extract_zip(dest, extract_dir)
        if not ex["ok"]:
            status = "EXTRACT_FAILED"
            message = ex["message"]
        else:
            extracted_size = int(ex.get("extracted_size_bytes") or dir_size(extract_dir))
            if ex["status"] == "EXISTS" and status == "EXISTS":
                message = "archive+extract already present"
            elif ex["status"] == "EXTRACTED":
                message = f"{message}; extracted"

        row = {
            "dataset": f"firesense_{meta['role']}",
            "status": status,
            "source_url": meta["url"],
            "destination": str(dest),
            "timestamp_utc": utc_now_iso(),
            "archive_or_repo": filename,
            "sha256": sha,
            "size_bytes": size,
            "extracted_size_bytes": extracted_size,
            "message": message,
            "extract_dir": str(extract_dir),
            "notes": meta["notes"],
        }
        append_csv(MANIFEST_DIR / "downloads.csv", row, DOWNLOAD_FIELDS)
        results.append(row)
    payload = {
        "status": "PASS" if results and all(r["status"] in {"DOWNLOADED", "EXISTS"} for r in results) else "PARTIAL_OR_FAIL",
        "timestamp_utc": utc_now_iso(),
        "archives": results,
        "video_inventory": inventory_firesense_videos(),
    }
    write_json(REPORT_DIR / "firesense_download_report.json", payload)
    write_md(REPORT_DIR / "firesense_download_report.md", "FIRESENSE Download Report", payload)
    return payload


def inventory_firesense_videos() -> dict:
    extract_root = FIRESENSE_DIR / "extracted"
    videos = list_videos(extract_root) if extract_root.exists() else []
    by_role = {"fire": [], "smoke": [], "other": []}
    for path in videos:
        rel = str(path.relative_to(extract_root)) if extract_root in path.parents or path.parent == extract_root else path.name
        low = rel.lower()
        if "smoke" in low:
            by_role["smoke"].append(rel)
        elif "fire" in low or "flame" in low:
            by_role["fire"].append(rel)
        else:
            by_role["other"].append(rel)
    return {
        "extract_root": str(extract_root),
        "video_count": len(videos),
        "by_role_counts": {k: len(v) for k, v in by_role.items()},
        "videos_sample": [str(p.relative_to(ROOT)) if ROOT in p.parents else str(p) for p in videos[:50]],
    }


def record_dfire_manual() -> dict:
    ensure_dirs(DFIRE_DIR, DFIRE_DIR / "presplit")
    record_license_dfire()
    rows = []
    for name, meta in DFIRE_MANUAL.items():
        print("MANUAL_DOWNLOAD_REQUIRED")
        print(f"dataset: {name}")
        print(f"official_url: {meta['url']}")
        print(f"destination: {meta['dest_hint']}")
        print(f"expected: {meta['expected']}")
        row = {
            "dataset": name,
            "status": "MANUAL_DOWNLOAD_REQUIRED",
            "source_url": meta["url"],
            "destination": meta["dest_hint"],
            "timestamp_utc": utc_now_iso(),
            "archive_or_repo": meta["expected"],
            "message": "See data/raw/dfire/MANUAL_DOWNLOAD.md (Kaggle official mirror preferred).",
            "priority": meta["priority"],
            "role": meta["role"],
            "object_detection_annotations": meta["od_annotations"],
        }
        append_csv(MANIFEST_DIR / "downloads.csv", row, DOWNLOAD_FIELDS)
        rows.append(row)
    payload = {
        "status": "MANUAL_DOWNLOAD_REQUIRED",
        "timestamp_utc": utc_now_iso(),
        "items": rows,
        "instructions": str(DFIRE_DIR / "MANUAL_DOWNLOAD.md"),
        "verify_hint": "python scripts/download_priority_sources.py verify",
    }
    write_json(REPORT_DIR / "dfire_manual_download_report.json", payload)
    write_md(REPORT_DIR / "dfire_manual_download_report.md", "D-Fire Manual Download Report", payload)
    return payload


def verify_local() -> dict:
    """Inventory what is already on disk for priority sources."""
    firesense_archives = {}
    for filename, meta in FIRESENSE_ARCHIVES.items():
        path = FIRESENSE_DIR / filename
        extract_dir = FIRESENSE_DIR / "extracted" / Path(filename).stem
        firesense_archives[filename] = {
            "archive_exists": path.exists(),
            "archive_bytes": path.stat().st_size if path.exists() else 0,
            "extract_exists": extract_dir.exists() and any(extract_dir.rglob("*")),
            "url": meta["url"],
            "role": meta["role"],
        }

    dfire_images = list_images(DFIRE_DIR)
    # Ignore the single example figure from the GitHub clone when judging readiness.
    meaningful_images = [
        p
        for p in dfire_images
        if "figures" not in {part.lower() for part in p.parts}
        and p.name.lower() != "dfire_examples.png"
    ]

    payload = {
        "timestamp_utc": utc_now_iso(),
        "firesense": {
            "archives": firesense_archives,
            "video_inventory": inventory_firesense_videos(),
            "ready_for_temporal_use": all(
                v["archive_exists"] and v["archive_bytes"] > 1_000_000 for v in firesense_archives.values()
            ),
            "object_detection_annotations": False,
        },
        "dfire_official": {
            "github_clone_present": (DFIRE_DIR / "DFireDataset").exists(),
            "image_count_total": len(dfire_images),
            "image_count_meaningful": len(meaningful_images),
            "images_ready_for_prepare_dfire": len(meaningful_images) >= 100,
            "presplit_ready": (DFIRE_DIR / "presplit").exists() and any((DFIRE_DIR / "presplit").rglob("*")),
            "manual_urls": {k: v["url"] for k, v in DFIRE_MANUAL.items()},
            "object_detection_annotations_when_images_present": True,
            "note": "Official D-Fire via Kaggle README mirror under kaggle_smoke_fire_detection_yolo/.",
        },
    }
    ready = []
    pending = []
    if payload["firesense"]["ready_for_temporal_use"]:
        ready.append("firesense_videos")
    else:
        pending.append("firesense_videos")
    if payload["dfire_official"]["images_ready_for_prepare_dfire"]:
        ready.append("dfire_images_labels")
    else:
        pending.append("dfire_images_labels")

    payload["summary"] = {
        "ready": ready,
        "pending": pending,
        "status": "READY" if not pending else ("PARTIAL" if ready else "PENDING"),
    }
    write_json(REPORT_DIR / "priority_sources_verify.json", payload)
    write_md(REPORT_DIR / "priority_sources_verify.md", "Priority Sources Verify", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["download", "verify", "record-manual", "all"],
        help="download=FIRESENSE ZIPs; record-manual=D-Fire ledger; verify=local inventory; all=download+manual+verify",
    )
    parser.add_argument("--firesense", action="store_true", help="With download: only FIRESENSE")
    parser.add_argument("--force", action="store_true", help="Re-download FIRESENSE even if archive exists")
    args = parser.parse_args()
    ensure_dirs(MANIFEST_DIR, REPORT_DIR, FIRESENSE_DIR, DFIRE_DIR)

    if args.command == "verify":
        payload = verify_local()
        print(json.dumps(payload["summary"], indent=2))
        return 0 if payload["summary"]["status"] in {"READY", "PARTIAL"} else 1

    if args.command == "record-manual":
        payload = record_dfire_manual()
        print(json.dumps({"status": payload["status"], "count": len(payload["items"])}, indent=2))
        return 0

    if args.command == "download":
        payload = download_firesense(force=args.force)
        print(json.dumps({"status": payload["status"], "archives": len(payload["archives"])}, indent=2))
        return 0 if payload["status"] == "PASS" else 2

    # all
    fire = download_firesense(force=args.force)
    manual = record_dfire_manual()
    verify = verify_local()
    summary = {
        "firesense": fire["status"],
        "dfire_manual": manual["status"],
        "verify": verify["summary"],
        "timestamp_utc": utc_now_iso(),
    }
    write_json(REPORT_DIR / "priority_sources_download_summary.json", summary)
    write_md(REPORT_DIR / "priority_sources_download_summary.md", "Priority Sources Download Summary", summary)
    print(json.dumps(summary, indent=2))
    return 0 if fire["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
