#!/usr/bin/env python3
"""Render frozen original/heatmap pairs for same-object double counting.

The heatmaps are existing CountGD outputs from run_countgd_batch.py.  This
script does not alter source pixels or rerun the model; it only validates the
saved counts and arranges the evidence into a report-ready figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--joined-csv", type=Path, required=True)
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


def load_rows(
    path: Path, relative_paths: set[str]
) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "relative_path",
            "class_name",
            "pred_count",
            "gt_count",
            "heatmap_path",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Joined CSV is missing columns: {sorted(missing)}")
        for row in reader:
            relative_path = str(row["relative_path"])
            if relative_path not in relative_paths:
                continue
            if relative_path in rows:
                raise RuntimeError(f"Duplicate joined row: {relative_path}")
            rows[relative_path] = row
    missing_rows = relative_paths - set(rows)
    if missing_rows:
        raise RuntimeError(f"Missing frozen joined rows: {sorted(missing_rows)}")
    return rows


def resolve_heatmap(joined_csv: Path, encoded_path: str) -> Path:
    path = Path(encoded_path)
    if not path.is_absolute():
        path = joined_csv.parent / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def fit_image(source: Image.Image, size: int) -> tuple[Image.Image, int, int]:
    image = source.copy()
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    return image, (size - image.width) // 2, (size - image.height) // 2


def render_panel(
    source: Image.Image,
    title: str,
    panel_size: int,
    header_height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    panel = Image.new("RGB", (panel_size, panel_size + header_height), "white")
    image, x_offset, y_inner = fit_image(source, panel_size)
    panel.paste(image, (x_offset, header_height + y_inner))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_size, header_height), fill=(20, 20, 20))
    draw.text((12, 12), title, fill=(255, 255, 255), font=font)
    return panel


def validate_example(
    example: dict[str, object], row: dict[str, str]
) -> None:
    actual = {
        "class_name": str(row["class_name"]),
        "requested_count": int(row["gt_count"]),
        "predicted_count": int(row["pred_count"]),
    }
    for field, value in actual.items():
        if example[field] != value:
            raise RuntimeError(
                f"Frozen value changed for {example['relative_path']} / {field}: "
                f"expected {example[field]}, got {value}"
            )


def main() -> None:
    args = parse_args()
    if args.panel_size < 400:
        raise ValueError("--panel-size must be at least 400")
    config = load_examples(args.examples)
    examples = config["examples"]
    relative_paths = {str(example["relative_path"]) for example in examples}
    rows = load_rows(args.joined_csv, relative_paths)

    header_height = max(52, args.panel_size // 11)
    label_height = max(48, args.panel_size // 12)
    panel_height = args.panel_size + header_height
    row_height = label_height + panel_height
    sheet = Image.new("RGB", (args.panel_size * 2, row_height * 2), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.load_default(size=max(18, args.panel_size // 28))
    label_font = ImageFont.load_default(size=max(17, args.panel_size // 30))

    for index, example in enumerate(examples):
        relative_path = str(example["relative_path"])
        row = rows[relative_path]
        validate_example(example, row)
        source_path = args.dataset_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        heatmap_path = resolve_heatmap(args.joined_csv, row["heatmap_path"])
        with Image.open(source_path) as handle:
            source = handle.convert("RGB")
        with Image.open(heatmap_path) as handle:
            heatmap = handle.convert("RGB")

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
            ),
            render_panel(
                heatmap,
                f"CountGD heatmap  |  predicted count = {example['predicted_count']}",
                args.panel_size,
                header_height,
                title_font,
            ),
        ]
        for column, panel in enumerate(panels):
            sheet.paste(
                panel,
                (column * args.panel_size, row_top + label_height),
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / str(config["output_file"])
    sheet.save(output_path, quality=95, subsampling=0)
    print(output_path)


if __name__ == "__main__":
    main()
