#!/usr/bin/env python3
"""Render frozen original/bounding-box pairs for same-object double counting.

The source pixels are never generated or retouched.  This script validates and
overlays detections saved by run_self_similarity_boxes.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CONFIDENCE_THRESHOLD = 0.4
BOX_COLOR = (255, 55, 70)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument(
        "--examples",
        type=Path,
        default=repo_root / "configs" / "self_similarity_examples.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "self_similarity_examples",
    )
    parser.add_argument("--panel-size", type=int, default=600)
    return parser.parse_args()


def load_examples(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    examples = config.get("examples", [])
    if len(examples) != 2:
        raise ValueError("The frozen self-similarity figure requires two examples")
    paths = [str(example["relative_path"]) for example in examples]
    if len(set(paths)) != len(paths):
        raise ValueError("Frozen examples contain duplicate relative paths")
    return config


def load_detections(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            relative_path = str(row["relative_path"])
            if relative_path in records:
                raise RuntimeError(f"Duplicate detection row: {relative_path}")
            records[relative_path] = row
    return records


def fit_image(source: Image.Image, size: int) -> tuple[Image.Image, int, int]:
    image = source.copy()
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    return image, (size - image.width) // 2, (size - image.height) // 2


def draw_detection(
    draw: ImageDraw.ImageDraw,
    item: dict[str, object],
    detection_index: int,
    image_size: tuple[int, int],
    offset: tuple[int, int],
    font: ImageFont.ImageFont,
) -> None:
    image_width, image_height = image_size
    x_offset, y_offset = offset
    cx = x_offset + float(item["cx"]) * image_width
    cy = y_offset + float(item["cy"]) * image_height
    width = float(item["w"]) * image_width
    height = float(item["h"]) * image_height
    left = cx - width / 2
    top = cy - height / 2
    right = cx + width / 2
    bottom = cy + height / 2
    stroke = max(4, image_width // 150)
    radius = max(12, image_width // 45)
    draw.rectangle((left, top, right, bottom), outline=BOX_COLOR, width=stroke)
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=BOX_COLOR,
        outline=(0, 0, 0),
        width=max(1, stroke // 2),
    )
    label = str(detection_index)
    label_box = draw.textbbox((0, 0), label, font=font)
    label_width = label_box[2] - label_box[0]
    label_height = label_box[3] - label_box[1]
    draw.text(
        (cx - label_width / 2, cy - label_height / 2 - 1),
        label,
        fill=(255, 255, 255),
        font=font,
    )


def render_panel(
    source: Image.Image,
    title: str,
    panel_size: int,
    header_height: int,
    title_font: ImageFont.ImageFont,
    box_font: ImageFont.ImageFont,
    detections: list[dict[str, object]] | None = None,
) -> Image.Image:
    panel = Image.new("RGB", (panel_size, panel_size + header_height), "white")
    image, x_offset, y_inner = fit_image(source, panel_size)
    y_offset = header_height + y_inner
    panel.paste(image, (x_offset, y_offset))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_size, header_height), fill=(20, 20, 20))
    draw.text((12, 12), title, fill=(255, 255, 255), font=title_font)
    for index, item in enumerate(detections or [], start=1):
        draw_detection(
            draw,
            item,
            index,
            image.size,
            (x_offset, y_offset),
            box_font,
        )
    return panel


def validate_record(
    example: dict[str, object], record: dict[str, object]
) -> list[dict[str, object]]:
    expected = {
        "class_name": str(example["class_name"]),
        "prompt": str(example["class_name"]),
        "requested_count": int(example["requested_count"]),
        "predicted_count": int(example["predicted_count"]),
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise RuntimeError(
                f"Frozen value changed for {example['relative_path']} / {field}: "
                f"expected {value!r}, got {record.get(field)!r}"
            )
    saved_threshold = float(record.get("min_saved_threshold", 1.0))
    if saved_threshold != CONFIDENCE_THRESHOLD:
        raise RuntimeError(
            f"Expected saved threshold {CONFIDENCE_THRESHOLD}, got {saved_threshold}"
        )
    detections = list(record["detections"])
    if len(detections) != int(example["predicted_count"]):
        raise RuntimeError(
            f"Detection count changed for {example['relative_path']}: "
            f"expected {example['predicted_count']}, got {len(detections)}"
        )
    if any(float(item["score"]) <= CONFIDENCE_THRESHOLD for item in detections):
        raise RuntimeError("Detection artifact contains a score at/below 0.4")
    return detections


def main() -> None:
    args = parse_args()
    if args.panel_size < 400:
        raise ValueError("--panel-size must be at least 400")
    config = load_examples(args.examples)
    examples = list(config["examples"])
    records = load_detections(args.detections)
    expected_paths = {str(example["relative_path"]) for example in examples}
    if set(records) != expected_paths:
        raise RuntimeError(
            "Detection rows do not match frozen examples: "
            f"expected {sorted(expected_paths)}, got {sorted(records)}"
        )

    header_height = max(52, args.panel_size // 11)
    label_height = max(48, args.panel_size // 12)
    panel_height = args.panel_size + header_height
    row_height = label_height + panel_height
    sheet = Image.new("RGB", (args.panel_size * 2, row_height * 2), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.load_default(size=max(18, args.panel_size // 28))
    label_font = ImageFont.load_default(size=max(17, args.panel_size // 30))
    box_font = ImageFont.load_default(size=max(16, args.panel_size // 32))

    for index, example in enumerate(examples):
        relative_path = str(example["relative_path"])
        detections = validate_record(example, records[relative_path])
        source_path = args.dataset_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        with Image.open(source_path) as handle:
            source = handle.convert("RGB")

        row_top = index * row_height
        label = (
            f"({chr(ord('a') + index)}) {Path(relative_path).name}: "
            f"{example['note']}"
        )
        draw.text((10, row_top + 12), label, fill=(0, 0, 0), font=label_font)
        panels = [
            render_panel(
                source,
                f"Original image  |  requested count = {example['requested_count']}",
                args.panel_size,
                header_height,
                title_font,
                box_font,
            ),
            render_panel(
                source,
                (
                    "CountGD bounding boxes  |  "
                    f"predicted count = {example['predicted_count']}"
                ),
                args.panel_size,
                header_height,
                title_font,
                box_font,
                detections,
            ),
        ]
        for column, panel in enumerate(panels):
            sheet.paste(panel, (column * args.panel_size, row_top + label_height))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / str(config["output_file"])
    sheet.save(output_path, quality=95, subsampling=0)
    print(output_path)


if __name__ == "__main__":
    main()
