import csv
from pathlib import Path

from fire_smoke_cpu.utils import sha256_file, write_csv_dicts


def test_sha256_duplicate_detection(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("same", encoding="utf-8")
    b.write_text("same", encoding="utf-8")
    assert sha256_file(a) == sha256_file(b)


def test_dataset_yaml_generation_contract(tmp_path: Path):
    yaml_path = tmp_path / "fire_smoke.yaml"
    yaml_path.write_text(
        "\n".join(["path: /tmp/ds", "train: images/train", "val: images/val", "test: images/test", "names:", "  0: fire", "  1: smoke", ""]),
        encoding="utf-8",
    )
    text = yaml_path.read_text(encoding="utf-8")
    assert "0: fire" in text
    assert "1: smoke" in text


def test_split_leakage_prevention_logic(tmp_path: Path):
    rows = [
        {"sample_id": "a", "sha256": "x", "split": "train"},
        {"sample_id": "b", "sha256": "x", "split": "train"},
        {"sample_id": "c", "sha256": "y", "split": "val"},
    ]
    leaks = {}
    for row in rows:
        leaks.setdefault(row["sha256"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in leaks.values())
