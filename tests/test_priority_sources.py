from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fire_smoke_cpu.provenance import validate_data_origin
import download_priority_sources as dps
import qualify_priority_sources as qps


def test_zenodo_and_official_provider_origins_allowed_for_real_training():
    assert validate_data_origin("zenodo", real_training=True).ok
    assert validate_data_origin("official_provider", real_training=True).ok
    assert not validate_data_origin("unknown", real_training=True).ok


def test_qualify_pending_when_archives_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dps, "FIRESENSE_DIR", tmp_path / "firesense")
    monkeypatch.setattr(dps, "DFIRE_DIR", tmp_path / "dfire")
    (tmp_path / "firesense").mkdir()
    (tmp_path / "dfire").mkdir()

    verify = {
        "firesense": {"ready_for_temporal_use": False},
        "dfire_official": {
            "images_ready_for_prepare_dfire": False,
            "presplit_ready": False,
        },
        "summary": {"ready": [], "pending": ["firesense_videos", "dfire_images_labels"], "status": "PENDING"},
    }
    payload = qps.qualify_from_verify(verify)
    assert payload["training_started"] is False
    assert payload["status"] == "PENDING_DOWNLOADS"
    decisions = {c["dataset"]: c["decision"] for c in payload["candidates"]}
    assert "badsaarow/d-fire" not in decisions
    assert "visifire" not in decisions
    assert "pyronear/pyro-sdis" not in decisions
    assert decisions["firesense_zenodo_836749"] == "ACCEPT_PENDING_DOWNLOAD"
    assert decisions["dfire_official_images_labels"] == "ACCEPT_PENDING_DOWNLOAD"


def test_qualify_temporal_pass_when_firesense_ready():
    verify = {
        "firesense": {"ready_for_temporal_use": True},
        "dfire_official": {
            "images_ready_for_prepare_dfire": False,
            "presplit_ready": False,
        },
        "summary": {"ready": ["firesense_videos"], "pending": ["dfire_images_labels"], "status": "PARTIAL"},
    }
    payload = qps.qualify_from_verify(verify)
    assert payload["status"] == "PASS_TEMPORAL_ONLY"
    assert payload["selected_for_od_training"] is None
    assert any(c["dataset"] == "firesense_zenodo_836749" for c in payload["selected_for_temporal"])


def test_qualify_od_ready_when_dfire_images_present():
    verify = {
        "firesense": {"ready_for_temporal_use": True},
        "dfire_official": {
            "images_ready_for_prepare_dfire": True,
            "presplit_ready": True,
        },
        "summary": {
            "ready": ["firesense_videos", "dfire_images_labels"],
            "pending": [],
            "status": "READY",
        },
    }
    payload = qps.qualify_from_verify(verify)
    assert payload["status"] == "PASS_OD_READY"
    assert payload["selected_for_od_training"]["dataset"] == "dfire_official_images_labels"
    assert payload["training_started"] is False
