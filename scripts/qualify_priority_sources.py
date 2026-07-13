#!/usr/bin/env python3
"""Qualify FIRESENSE + official D-Fire for SecureVU (no training).

Runs verify inventory, historical-registry rebuild, and a dedicated qualification
report. Does not train, does not overwrite V2.1, and does not promote challengers.

Stop conditions:
  - FIRESENSE: ACCEPT for temporal/eval only (no YOLO boxes yet)
  - Official D-Fire images: ACCEPT when prepared locally
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fire_smoke_cpu.constants import MANIFEST_DIR, REPORT_DIR
from fire_smoke_cpu.utils import ensure_dirs, utc_now_iso

import download_priority_sources as dps
import error_driven_workflow as edw


def _candidate(
    *,
    name: str,
    url: str,
    local_path: Path,
    license_status: str,
    od_annotations: bool,
    decision: str,
    notes: str,
    role: str,
    reject_reasons: list[str] | None = None,
    local_exists: bool | None = None,
) -> dict:
    exists = local_path.exists() and any(local_path.rglob("*")) if local_exists is None else local_exists
    return {
        "dataset": name,
        "url": url,
        "local_path": str(local_path.relative_to(ROOT)) if ROOT in local_path.parents else str(local_path),
        "local_exists": exists,
        "license_status": license_status,
        "object_detection_annotations": od_annotations,
        "decision": decision,
        "reject_reasons": reject_reasons or [],
        "notes": notes,
        "role": role,
    }


def qualify_from_verify(verify: dict) -> dict:
    fs = verify["firesense"]
    df = verify["dfire_official"]

    firesense_ready = bool(fs.get("ready_for_temporal_use"))
    dfire_images_ready = bool(df.get("images_ready_for_prepare_dfire"))
    dfire_presplit_ready = bool(df.get("presplit_ready"))

    candidates = [
        _candidate(
            name="firesense_zenodo_836749",
            url="https://zenodo.org/records/836749",
            local_path=dps.FIRESENSE_DIR,
            license_status="requires_review",
            od_annotations=False,
            decision="ACCEPT_TEMPORAL_ONLY" if firesense_ready else "ACCEPT_PENDING_DOWNLOAD",
            notes=(
                "Official FIRESENSE fire+smoke video ZIPs. Video-only; not YOLO OD. "
                "Use for temporal/eval clips after provenance review. Do not blind-merge into V2.1 train."
            ),
            role="temporal_eval_videos",
            local_exists=firesense_ready or (dps.FIRESENSE_DIR.exists() and any(dps.FIRESENSE_DIR.glob("*.zip"))),
            reject_reasons=[] if firesense_ready else ["archive_not_ready"],
        ),
        _candidate(
            name="dfire_official_images_labels",
            url=dps.DFIRE_MANUAL["dfire_images_labels"]["url"],
            local_path=dps.DFIRE_DIR,
            license_status="CC0-1.0",
            od_annotations=True,
            decision="ACCEPT" if dfire_images_ready else "ACCEPT_PENDING_DOWNLOAD",
            notes=(
                "Official D-Fire images + YOLO labels (Kaggle README mirror). "
                "Primary detector training candidate via scripts/prepare_dfire.py."
            ),
            role="detector_training",
            local_exists=dfire_images_ready,
            reject_reasons=[] if dfire_images_ready else ["official_images_not_local"],
        ),
        _candidate(
            name="dfire_official_presplit",
            url=dps.DFIRE_MANUAL["dfire_presplit"]["url"],
            local_path=dps.DFIRE_DIR / "presplit",
            license_status="CC0-1.0",
            od_annotations=True,
            decision="ACCEPT" if dfire_presplit_ready else "ACCEPT_PENDING_DOWNLOAD",
            notes="Train/val/test splits from the official Kaggle D-Fire mirror.",
            role="detector_training_presplit",
            local_exists=dfire_presplit_ready,
            reject_reasons=[] if dfire_presplit_ready else ["presplit_not_local"],
        ),
    ]

    accepted = [
        c
        for c in candidates
        if c["decision"] in {"ACCEPT", "ACCEPT_TEMPORAL_ONLY", "ACCEPT_PENDING_DOWNLOAD"}
    ]
    ready_od = [c for c in candidates if c["decision"] == "ACCEPT" and c["object_detection_annotations"]]
    ready_temporal = [c for c in candidates if c["decision"] == "ACCEPT_TEMPORAL_ONLY"]

    if ready_od:
        status = "PASS_OD_READY"
    elif ready_temporal:
        status = "PASS_TEMPORAL_ONLY"
    elif any(c["decision"] == "ACCEPT_PENDING_DOWNLOAD" for c in candidates):
        status = "PENDING_DOWNLOADS"
    else:
        status = "NO_GO"

    next_actions = []
    if not firesense_ready:
        next_actions.append("python scripts/download_priority_sources.py download --firesense")
    if not dfire_images_ready:
        next_actions.append("Follow data/raw/dfire/MANUAL_DOWNLOAD.md then: python scripts/download_priority_sources.py verify")
    if dfire_images_ready:
        next_actions.append("python scripts/prepare_dfire.py")
        next_actions.append("python scripts/error_driven_workflow.py historical-registry")
        next_actions.append("# Later (still no blind merge): audit/overlap on official dfire, then frozen V2.1 error-analyze")
    next_actions.append("Do NOT run train-1e until quality gates pass on hard-example selection")

    return {
        "status": status,
        "timestamp_utc": utc_now_iso(),
        "training_started": False,
        "v2_1_overwrite": False,
        "selected_for_od_training": ready_od[0] if ready_od else None,
        "selected_for_temporal": ready_temporal,
        "candidates": candidates,
        "accepted_count": len(accepted),
        "verify_summary": verify.get("summary"),
        "next_actions": next_actions,
        "policy": {
            "blind_merges_prohibited": True,
            "blocked_boosters": sorted(edw.BLOCKED_NEW_SOURCES),
            "require_historical_overlap_before_train": True,
            "require_frozen_v2_1_error_analyze_before_train": True,
        },
    }


def run() -> dict:
    ensure_dirs(REPORT_DIR, MANIFEST_DIR)
    verify = dps.verify_local()
    hist = edw.build_historical_registry()
    payload = qualify_from_verify(verify)
    payload["historical_registry"] = {
        "total_rows": hist.get("total_rows"),
        "unique_sha256": hist.get("unique_sha256"),
        "status": hist.get("status"),
    }
    dps.write_json(REPORT_DIR / "priority_sources_qualification.json", payload)
    dps.write_md(REPORT_DIR / "priority_sources_qualification.md", "Priority Sources Qualification", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()
    payload = run()
    if args.json_only:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps({
            "status": payload["status"],
            "selected_for_od_training": payload["selected_for_od_training"],
            "selected_for_temporal": [c["dataset"] for c in payload["selected_for_temporal"]],
            "next_actions": payload["next_actions"],
            "training_started": False,
        }, indent=2))
    return 0 if payload["status"] in {"PASS_OD_READY", "PASS_TEMPORAL_ONLY", "PENDING_DOWNLOADS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
