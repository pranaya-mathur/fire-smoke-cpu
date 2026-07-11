from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def run_command(cmd: list[str], timeout: int = 30) -> dict:
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "cmd": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - defensive hardware probes
        return {"cmd": cmd, "error": repr(exc)}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image.load()
    return image


def image_size(path: Path) -> tuple[int, int]:
    with open_image(path) as image:
        return image.size


def perceptual_hash(path: Path) -> str:
    import imagehash

    with open_image(path) as image:
        return str(imagehash.phash(image.convert("RGB")))


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def list_videos(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def write_json(path: Path, data: object) -> None:
    ensure_dirs(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv_dicts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_dicts(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def append_csv_dicts(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    rows = list(rows)
    if not rows:
        return
    ensure_dirs(path.parent)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def safe_link_or_copy(src: Path, dst: Path) -> str:
    ensure_dirs(dst.parent)
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        try:
            dst.symlink_to(src)
            return "symlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"


def git_commit() -> str:
    result = run_command(["git", "rev-parse", "HEAD"])
    if result.get("returncode") == 0:
        return result.get("stdout", "")
    return "not_available"


def python_info() -> dict:
    return {
        "version": platform.python_version(),
        "executable": shutil.which("python") or "",
        "implementation": platform.python_implementation(),
        "architecture": platform.machine(),
    }
