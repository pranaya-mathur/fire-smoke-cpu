from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .constants import CLASS_NAME_TO_ID


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def as_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.8f} {self.y_center:.8f} "
            f"{self.width:.8f} {self.height:.8f}"
        )


def canonical_class_id(name: str) -> int | None:
    return CLASS_NAME_TO_ID.get(name.strip().lower())


def validate_yolo_box(box: YoloBox) -> list[str]:
    errors: list[str] = []
    if box.class_id not in {0, 1}:
        errors.append("unknown_class_id")
    values = [box.x_center, box.y_center, box.width, box.height]
    if not all(math.isfinite(v) for v in values):
        errors.append("non_finite_coordinate")
    if box.width <= 0 or box.height <= 0:
        errors.append("zero_or_negative_area")
    if not all(0 <= v <= 1 for v in values):
        errors.append("coordinate_out_of_range")
    left = box.x_center - box.width / 2
    right = box.x_center + box.width / 2
    top = box.y_center - box.height / 2
    bottom = box.y_center + box.height / 2
    if left < -1e-6 or top < -1e-6 or right > 1 + 1e-6 or bottom > 1 + 1e-6:
        errors.append("box_outside_image")
    return errors


def parse_yolo_label(path: Path) -> tuple[list[YoloBox], list[str]]:
    boxes: list[YoloBox] = []
    errors: list[str] = []
    if not path.exists():
        return boxes, ["missing_label"]
    for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line_{line_no}:wrong_field_count")
            continue
        try:
            box = YoloBox(int(float(parts[0])), *(float(v) for v in parts[1:]))
        except ValueError:
            errors.append(f"line_{line_no}:parse_error")
            continue
        box_errors = validate_yolo_box(box)
        errors.extend(f"line_{line_no}:{err}" for err in box_errors)
        boxes.append(box)
    return boxes, errors


def voc_box_to_yolo(
    class_id: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int,
) -> YoloBox:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("invalid bbox with non-positive area")
    x_center = ((xmin + xmax) / 2.0) / image_width
    y_center = ((ymin + ymax) / 2.0) / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    return YoloBox(class_id, x_center, y_center, width, height)


def parse_voc_xml(path: Path, fallback_size: tuple[int, int] | None = None) -> tuple[list[YoloBox], list[str], dict]:
    errors: list[str] = []
    metadata: dict = {"unknown_classes": []}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return [], [f"xml_parse_error:{exc}"], metadata

    width = root.findtext("size/width")
    height = root.findtext("size/height")
    try:
        image_width = int(float(width)) if width else 0
        image_height = int(float(height)) if height else 0
    except ValueError:
        image_width, image_height = 0, 0
    if (image_width <= 0 or image_height <= 0) and fallback_size:
        image_width, image_height = fallback_size
    if image_width <= 0 or image_height <= 0:
        errors.append("invalid_or_missing_image_size")

    boxes: list[YoloBox] = []
    for index, obj in enumerate(root.findall("object"), start=1):
        name = obj.findtext("name") or ""
        class_id = canonical_class_id(name)
        if class_id is None:
            metadata["unknown_classes"].append(name)
            errors.append(f"object_{index}:unknown_class:{name}")
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            errors.append(f"object_{index}:missing_bndbox")
            continue
        try:
            xmin = float(bnd.findtext("xmin", "nan"))
            ymin = float(bnd.findtext("ymin", "nan"))
            xmax = float(bnd.findtext("xmax", "nan"))
            ymax = float(bnd.findtext("ymax", "nan"))
            box = voc_box_to_yolo(class_id, xmin, ymin, xmax, ymax, image_width, image_height)
        except Exception as exc:
            errors.append(f"object_{index}:invalid_bbox:{exc}")
            continue
        box_errors = validate_yolo_box(box)
        if box_errors:
            errors.extend(f"object_{index}:{err}" for err in box_errors)
        boxes.append(box)
    return boxes, errors, metadata


def write_yolo_label(path: Path, boxes: list[YoloBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(box.as_line() for box in boxes) + ("\n" if boxes else ""), encoding="utf-8")
