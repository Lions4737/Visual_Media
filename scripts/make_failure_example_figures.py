#!/usr/bin/env python3
"""Render frozen before/auxiliary/after panels from saved CountGD boxes.

The source pixels are never generated or retouched.  This script only resizes
the original image and overlays the detections stored by run_clock_prompts.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

if __package__:
    from .analyze_prompt_union import (
        CONFIDENCE_THRESHOLD,
        IOU_THRESHOLD,
        PROMPTS,
        selected,
        spatial_union,
    )
else:
    from analyze_prompt_union import (
        CONFIDENCE_THRESHOLD,
        IOU_THRESHOLD,
        PROMPTS,
        selected,
        spatial_union,
    )


BEFORE_COLOR = (255, 65, 80)
AUXILIARY_COLOR = (65, 125, 255)
AFTER_COLOR = (35, 210, 115)
ADDED_COLOR = (255, 205, 45)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--raw",
        type=Path,
        action="append",
        required=True,
        help="prompt-detection JSONL; repeat when runs are split across files",
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=repo_root / "configs" / "failure_examples.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "failure_examples",
    )
    parser.add_argument("--panel-size", type=int, default=500)
    return parser.parse_args()


def load_records(paths: list[Path]) -> dict[tuple[str, str], dict[str, object]]:
    records: dict[tuple[str, str], dict[str, object]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt_id = str(row["prompt_id"])
                if prompt_id not in PROMPTS:
                    continue
                if str(row.get("prompt")) != PROMPTS[prompt_id]:
                    raise RuntimeError(
                        f"Prompt text does not match frozen id {prompt_id!r}: "
                        f"{row.get('prompt')!r}"
                    )
                if (
                    float(row.get("min_saved_threshold", 0.0))
                    > CONFIDENCE_THRESHOLD
                ):
                    raise RuntimeError(
                        "Raw detections were truncated above the fixed figure "
                        f"threshold for {row.get('relative_path')} / {prompt_id}"
                    )
                key = (str(row["relative_path"]), prompt_id)
                if key in records:
                    raise RuntimeError(f"Duplicate prompt row: {key}")
                records[key] = row
    return records


def load_examples(path: Path) -> dict[str, dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    expected_groups = {"failure_condition_1", "failure_condition_3"}
    if set(config) != expected_groups:
        raise ValueError(f"Expected frozen groups: {sorted(expected_groups)}")
    for group_name, group in config.items():
        examples = group.get("examples", [])
        if len(examples) != 2:
            raise ValueError(f"{group_name} must contain exactly two examples")
    return config


def fit_image(source: Image.Image, size: int) -> tuple[Image.Image, int, int]:
    image = source.copy()
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    return image, (size - image.width) // 2, (size - image.height) // 2


def draw_detection(
    draw: ImageDraw.ImageDraw,
    item: dict[str, object],
    color: tuple[int, int, int],
    image_size: tuple[int, int],
    offset: tuple[int, int],
) -> None:
    image_width, image_height = image_size
    x_offset, y_offset = offset
    cx = x_offset + float(item["cx"]) * image_width
    cy = y_offset + float(item["cy"]) * image_height
    width = float(item["w"]) * image_width
    height = float(item["h"]) * image_height
    box = (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
    stroke = max(3, image_width // 180)
    radius = max(5, image_width // 90)
    draw.rectangle(box, outline=color, width=stroke)
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=color,
        outline=(0, 0, 0),
        width=max(1, stroke // 2),
    )


def render_panel(
    source: Image.Image,
    detections: list[dict[str, object]],
    title: str,
    base_color: tuple[int, int, int],
    panel_size: int,
    font: ImageFont.ImageFont,
    highlight_added: bool = False,
) -> Image.Image:
    header = max(40, panel_size // 12)
    panel = Image.new("RGB", (panel_size, panel_size + header), "white")
    image, x_offset, y_inner = fit_image(source, panel_size)
    y_offset = header + y_inner
    panel.paste(image, (x_offset, y_offset))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_size, header), fill=(20, 20, 20))
    draw.text((10, 9), title, fill=(255, 255, 255), font=font)
    for item in detections:
        color = base_color
        if highlight_added and set(item.get("prompts", [])) == {"digital clock"}:
            color = ADDED_COLOR
        draw_detection(
            draw,
            item,
            color,
            image.size,
            (x_offset, y_offset),
        )
    return panel


def validate_counts(
    example: dict[str, object],
    before: list[dict[str, object]],
    auxiliary: list[dict[str, object]],
    after: list[dict[str, object]],
    canonical_row: dict[str, object],
    digital_row: dict[str, object],
) -> None:
    canonical_gt = int(canonical_row["gt_count"])
    digital_gt = int(digital_row["gt_count"])
    if canonical_gt != digital_gt:
        raise RuntimeError(
            f"Prompt rows disagree on requested count for {example['relative_path']}: "
            f"clock={canonical_gt}, digital clock={digital_gt}"
        )
    actual = {
        "requested_count": canonical_gt,
        "before_count": len(before),
        "digital_count": len(auxiliary),
        "after_count": len(after),
    }
    for field, value in actual.items():
        if int(example[field]) != value:
            raise RuntimeError(
                f"Frozen count changed for {example['relative_path']} / {field}: "
                f"expected {example[field]}, got {value}"
            )


def render_group(
    group: dict[str, object],
    records: dict[tuple[str, str], dict[str, object]],
    dataset_root: Path,
    output_dir: Path,
    panel_size: int,
) -> Path:
    font = ImageFont.load_default(size=max(14, panel_size // 30))
    label_font = ImageFont.load_default(size=max(13, panel_size // 34))
    row_label_height = max(36, panel_size // 12)
    panel_height = panel_size + max(40, panel_size // 12)
    row_height = row_label_height + panel_height
    sheet = Image.new("RGB", (panel_size * 3, row_height * 2), "white")
    sheet_draw = ImageDraw.Draw(sheet)

    for row_index, example in enumerate(group["examples"]):
        relative_path = str(example["relative_path"])
        image_path = dataset_root / relative_path
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as handle:
            source = handle.convert("RGB")

        canonical_row = records[(relative_path, "canonical")]
        digital_row = records[(relative_path, "digital")]
        before = selected(canonical_row)
        auxiliary = selected(digital_row)
        after = spatial_union(
            {"canonical": before, "digital": auxiliary}, IOU_THRESHOLD
        )
        validate_counts(
            example,
            before,
            auxiliary,
            after,
            canonical_row,
            digital_row,
        )

        row_top = row_index * row_height
        short_name = Path(relative_path).name
        label = (
            f"({chr(ord('a') + row_index)}) {short_name}: "
            f"request={example['requested_count']} | {example['note']}"
        )
        sheet_draw.text((8, row_top + 8), label, fill=(0, 0, 0), font=label_font)
        panels = [
            render_panel(
                source,
                before,
                f"Before: clock  count={len(before)}",
                BEFORE_COLOR,
                panel_size,
                font,
            ),
            render_panel(
                source,
                auxiliary,
                f"Auxiliary: digital clock  count={len(auxiliary)}",
                AUXILIARY_COLOR,
                panel_size,
                font,
            ),
            render_panel(
                source,
                after,
                (
                    f"After: spatial union  count={len(after)}"
                    + (
                        "  (yellow=digital-only)"
                        if any(
                            set(item.get("prompts", [])) == {"digital clock"}
                            for item in after
                        )
                        else ""
                    )
                ),
                AFTER_COLOR,
                panel_size,
                font,
                highlight_added=True,
            ),
        ]
        for column, panel in enumerate(panels):
            sheet.paste(panel, (column * panel_size, row_top + row_label_height))

    output = output_dir / str(group["output_file"])
    sheet.save(output, quality=95, subsampling=0)
    return output


def main() -> None:
    args = parse_args()
    if args.panel_size < 300:
        raise ValueError("--panel-size must be at least 300")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.raw)
    config = load_examples(args.examples)
    for group in config.values():
        output = render_group(
            group,
            records,
            args.dataset_root,
            args.output_dir,
            args.panel_size,
        )
        print(output)


if __name__ == "__main__":
    main()
