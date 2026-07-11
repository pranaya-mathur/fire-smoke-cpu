#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_smoke_cpu.constants import MANIFEST_DIR, RAW_DIR
from fire_smoke_cpu.utils import ensure_dirs, sha256_file, utc_now_iso


DATASETS = {
    "dfire": {
        "url": "https://github.com/gaia-solutions-on-demand/DFireDataset",
        "data_url": "https://1drv.ms/u/c/c0bd25b6b048b01d/EbLgD7bES4FDvUN37Grxn8QBF5gIBBc7YV2qklF08GCiBw",
        "split_url": "https://1drv.ms/f/c/c0bd25b6b048b01d/Ema8FFze8mFIlM1Hn81BUUgBE3vnnmK4SQxybS-nHRt2pA?e=6rk0aN",
        "dest": RAW_DIR / "dfire",
        "expected": "official D-Fire images/labels archive from OneDrive plus optional split archive",
        "license": "CC0-1.0 discovered in official GitHub LICENSE",
        "commercial_use_status": "confirmed by license text; still review in commercial release process",
        "auto": "git",
    },
    "ms_fsdb": {
        "url": "https://drive.google.com/file/d/14ylxaNBVmXjAFXt2h4lnyBe7xELhOVHc/view?usp=drive_link",
        "dest": RAW_DIR / "ms_fsdb",
        "expected": "official MS-FSDB archive from Google Drive",
        "license": "UNKNOWN",
        "commercial_use_status": "requires review",
        "auto": "manual_drive",
    },
    "mivia": {
        "url": "https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/",
        "dest": RAW_DIR / "mivia",
        "expected": "MIVIA fire detection videos after any required license/download flow",
        "license": "UNKNOWN",
        "commercial_use_status": "requires review",
        "auto": "manual_web",
    },
    "fasdd": {
        "url": "https://doi.org/10.57760/sciencedb.j00104.00103",
        "dest": RAW_DIR / "fasdd",
        "expected": "FASDD archive from ScienceDB DOI landing page",
        "license": "UNKNOWN",
        "commercial_use_status": "requires review",
        "auto": "manual_web",
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
LICENSE_FIELDS = ["name", "source_url", "license_discovered", "license_file_url", "commercial_use_status", "notes"]


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def append(path: Path, row: dict, fields: list[str]) -> None:
    ensure_dirs(path.parent)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def record_manual(name: str, info: dict) -> None:
    print("MANUAL_DOWNLOAD_REQUIRED")
    print(f"dataset: {name}")
    print(f"official_url: {info['url']}")
    print(f"destination: {info['dest']}")
    print(f"expected_archive: {info['expected']}")
    append(
        MANIFEST_DIR / "downloads.csv",
        {
            "dataset": name,
            "status": "MANUAL_DOWNLOAD_REQUIRED",
            "source_url": info["url"],
            "destination": str(info["dest"]),
            "timestamp_utc": utc_now_iso(),
            "archive_or_repo": info["expected"],
            "message": "Official provider requires browser/license/access flow; not bypassed.",
        },
        DOWNLOAD_FIELDS,
    )


def maybe_extract_archives(dest: Path) -> list[str]:
    extracted = []
    for archive in sorted(dest.glob("*")):
        if archive.suffix.lower() == ".zip":
            out_dir = dest / archive.stem
            if out_dir.exists():
                continue
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(out_dir)
            extracted.append(str(out_dir))
    return extracted


def download_dfire(info: dict) -> None:
    dest = info["dest"]
    ensure_dirs(dest)
    repo_dir = dest / "DFireDataset"
    if repo_dir.exists():
        status = "EXISTS"
        message = "Existing clone/archive retained."
    else:
        git = shutil.which("git")
        if not git:
            record_manual("dfire", info)
            return
        print(f"Running git clone from {info['url']} into {repo_dir}; resumable by preserving existing directory.")
        import subprocess

        result = subprocess.run([git, "clone", "--depth", "1", info["url"], str(repo_dir)], text=True, capture_output=True)
        status = "DOWNLOADED" if result.returncode == 0 else "MANUAL_DOWNLOAD_REQUIRED"
        message = result.stderr.strip() or result.stdout.strip()
        if result.returncode != 0:
            print(message)
            record_manual("dfire", info)
            return
    append(
        MANIFEST_DIR / "downloads.csv",
        {
            "dataset": "dfire",
            "status": status,
            "source_url": info["url"],
            "destination": str(dest),
            "timestamp_utc": utc_now_iso(),
            "archive_or_repo": str(repo_dir),
            "size_bytes": dir_size(repo_dir),
            "extracted_size_bytes": dir_size(repo_dir),
            "message": message,
        },
        DOWNLOAD_FIELDS,
    )
    image_count = len([p for p in repo_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    if image_count <= 1:
        print("MANUAL_DOWNLOAD_REQUIRED")
        print("dataset: dfire_images")
        print(f"official_url: {info['data_url']}")
        print(f"official_split_url: {info['split_url']}")
        print(f"destination: {dest}")
        print(f"expected_archive: {info['expected']}")
        append(
            MANIFEST_DIR / "downloads.csv",
            {
                "dataset": "dfire_images",
                "status": "MANUAL_DOWNLOAD_REQUIRED",
                "source_url": info["data_url"],
                "destination": str(dest),
                "timestamp_utc": utc_now_iso(),
                "archive_or_repo": info["expected"],
                "message": "Official GitHub repo cloned, but actual images/labels are hosted through OneDrive and require manual download if automated access is blocked.",
            },
            DOWNLOAD_FIELDS,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*DATASETS.keys(), "all"], required=True)
    args = parser.parse_args()
    ensure_dirs(MANIFEST_DIR)
    targets = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    for name in targets:
        info = DATASETS[name]
        ensure_dirs(info["dest"])
        append(
            MANIFEST_DIR / "licenses.csv",
            {
                "name": name,
                "source_url": info["url"],
                "license_discovered": info["license"],
                "commercial_use_status": info["commercial_use_status"],
                "notes": "Initial source tracking only; license requires human/legal review.",
            },
            LICENSE_FIELDS,
        )
        if info["auto"] == "git":
            download_dfire(info)
        else:
            maybe_extract_archives(info["dest"])
            record_manual(name, info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
