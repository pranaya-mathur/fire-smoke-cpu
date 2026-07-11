from pathlib import Path

import pytest

from fire_smoke_cpu.annotations import YoloBox, canonical_class_id, parse_yolo_label, validate_yolo_box, voc_box_to_yolo, write_yolo_label


def test_pascal_voc_to_yolo_conversion():
    box = voc_box_to_yolo(0, 10, 20, 30, 60, 100, 100)
    assert box.class_id == 0
    assert box.x_center == pytest.approx(0.2)
    assert box.y_center == pytest.approx(0.4)
    assert box.width == pytest.approx(0.2)
    assert box.height == pytest.approx(0.4)


def test_canonical_class_mapping_case_insensitive():
    assert canonical_class_id("Fire") == 0
    assert canonical_class_id(" smoke ") == 1
    assert canonical_class_id("flame") is None


def test_invalid_bounding_box_detection():
    errors = validate_yolo_box(YoloBox(0, 0.5, 0.5, 0.0, 0.2))
    assert "zero_or_negative_area" in errors


def test_empty_negative_label_handling(tmp_path: Path):
    label = tmp_path / "negative.txt"
    write_yolo_label(label, [])
    boxes, errors = parse_yolo_label(label)
    assert boxes == []
    assert errors == []
