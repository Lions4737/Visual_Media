#!/usr/bin/env python3
"""Freeze a final clock evaluation manifest after prompt selection.

The final set contains:
- every digital-mixed clock image not used by discovery/heldout prompt sweeps;
- five baseline-exact analog-only controls per requested count (40 total).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

NUMBER_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=repo_root
        / "configs"
        / "annotations"
        / "clock_visual_annotations.csv",
    )
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        default=repo_root / "configs" / "manifests" / "clock_prompt_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "manifests",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_rows(
    rows: list[dict[str, str]], required: set[str], label: str
) -> None:
    if not rows:
        raise ValueError(f"{label} is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    paths = [row["relative_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{label} contains duplicate relative_path values")


def gt_from_path(relative_path: str) -> int:
    return NUMBER_WORDS[Path(relative_path).parts[0]]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "clock_final_eval_manifest.csv"
    remaining_analog_output = (
        args.output_dir / "clock_remaining_analog_manifest.csv"
    )
    annotations = read_rows(args.annotations)
    validate_rows(
        annotations,
        {"relative_path", "has_digital", "baseline_error_type"},
        "annotations",
    )
    invalid_labels = {
        row["has_digital"] for row in annotations if row["has_digital"] not in {"0", "1"}
    }
    if invalid_labels:
        raise ValueError(f"Invalid has_digital values: {sorted(invalid_labels)}")
    pilot_rows = read_rows(args.pilot_manifest)
    validate_rows(
        pilot_rows,
        {"split", "failure_type", "gt_count", "relative_path"},
        "pilot manifest",
    )
    annotation_paths = {row["relative_path"] for row in annotations}
    used = {row["relative_path"] for row in pilot_rows}
    unknown_pilot_paths = used - annotation_paths
    if unknown_pilot_paths:
        raise ValueError(
            "Pilot manifest paths are missing from annotations: "
            f"{sorted(unknown_pilot_paths)[:10]}"
        )
    for row in pilot_rows:
        if int(row["gt_count"]) != gt_from_path(row["relative_path"]):
            raise ValueError(
                f"Pilot gt_count disagrees with path: {row['relative_path']}"
            )

    final_rows: list[dict[str, object]] = []
    for row in annotations:
        if row["relative_path"] in used or row["has_digital"] != "1":
            continue
        final_rows.append(
            {
                "split": "final_eval",
                "failure_type": "digital_mixed_remaining",
                "gt_count": gt_from_path(row["relative_path"]),
                "relative_path": row["relative_path"],
            }
        )

    analog_exact = [
        row
        for row in annotations
        if row["relative_path"] not in used
        and row["has_digital"] == "0"
        and row["baseline_error_type"] == "exact"
    ]
    for gt_count in range(2, 10):
        group = sorted(
            (
                row
                for row in analog_exact
                if gt_from_path(row["relative_path"]) == gt_count
            ),
            key=lambda row: row["relative_path"],
        )
        if len(group) < 5:
            raise RuntimeError(f"Not enough analog controls for count={gt_count}")
        for row in group[:5]:
            final_rows.append(
                {
                    "split": "final_eval",
                    "failure_type": "analog_exact_control",
                    "gt_count": gt_count,
                    "relative_path": row["relative_path"],
                }
            )

    final_rows.sort(
        key=lambda row: (
            str(row["failure_type"]),
            int(row["gt_count"]),
            str(row["relative_path"]),
        )
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "failure_type", "gt_count", "relative_path"],
        )
        writer.writeheader()
        writer.writerows(final_rows)

    mixed = sum(row["failure_type"] == "digital_mixed_remaining" for row in final_rows)
    controls = sum(row["failure_type"] == "analog_exact_control" for row in final_rows)
    print(f"Wrote {output}: mixed={mixed}, analog_controls={controls}")

    all_used = used | {str(row["relative_path"]) for row in final_rows}
    remaining_analog = [
        {
            "split": "remaining_analog",
            "failure_type": "analog_only_remaining",
            "gt_count": gt_from_path(row["relative_path"]),
            "relative_path": row["relative_path"],
        }
        for row in annotations
        if row["has_digital"] == "0" and row["relative_path"] not in all_used
    ]
    remaining_analog.sort(
        key=lambda row: (int(row["gt_count"]), str(row["relative_path"]))
    )
    with remaining_analog_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "failure_type", "gt_count", "relative_path"],
        )
        writer.writeheader()
        writer.writerows(remaining_analog)
    print(f"Wrote {remaining_analog_output}: samples={len(remaining_analog)}")


if __name__ == "__main__":
    main()
