from pathlib import Path
import runpy

import pytest

from fire_smoke_cpu.annotations import parse_yolo_label, write_yolo_label
from fire_smoke_cpu.provenance import assert_real_training_origins, validate_data_origin


def test_real_training_rejects_mock_placeholder_and_unknown_origins():
    rows = [
        {"sample_id": "ok", "data_origin": "huggingface_snapshot"},
        {"sample_id": "bad_mock", "data_origin": "mock"},
        {"sample_id": "bad_placeholder", "data_origin": "placeholder"},
        {"sample_id": "bad_unknown", "data_origin": "unknown"},
    ]
    with pytest.raises(ValueError) as exc:
        assert_real_training_origins(rows)
    text = str(exc.value)
    assert "bad_mock" in text
    assert "bad_placeholder" in text
    assert "bad_unknown" in text


def test_local_existing_and_huggingface_snapshot_are_allowed_for_real_training():
    rows = [
        {"sample_id": "hf", "data_origin": "huggingface_snapshot"},
        {"sample_id": "local", "data_origin": "local_existing"},
    ]
    assert_real_training_origins(rows)


def test_unrecognized_origin_is_not_silently_allowed():
    decision = validate_data_origin("internet_download", real_training=True)
    assert not decision.ok
    assert "unrecognized_data_origin" in decision.reason


def test_missing_label_is_not_confirmed_negative(tmp_path: Path):
    missing = tmp_path / "missing.txt"
    boxes, errors = parse_yolo_label(missing)
    assert boxes == []
    assert errors == ["missing_label"]


def test_empty_label_is_confirmed_negative(tmp_path: Path):
    label = tmp_path / "empty.txt"
    write_yolo_label(label, [])
    boxes, errors = parse_yolo_label(label)
    assert boxes == []
    assert errors == []


def test_main_training_epoch_ceiling_constant():
    root = Path(__file__).resolve().parents[1]
    module_globals = runpy.run_path(str(root / "scripts/real_v2_1_workflow.py"), run_name="not_main")

    assert module_globals["MAX_MAIN_EPOCHS"] == 12
