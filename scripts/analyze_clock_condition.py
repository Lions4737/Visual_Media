#!/usr/bin/env python3
"""Quantify the manually audited digital-clock failure condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import fisher_exact


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined-csv", type=Path, required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=repo_root
        / "configs"
        / "annotations"
        / "clock_visual_annotations.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "clock_condition",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.joined_csv)
    required_columns = {
        "relative_path",
        "class_name",
        "gt_count",
        "failure",
        "under",
        "over",
        "abs_error",
        "signed_error",
    }
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Joined CSV is missing columns: {sorted(missing_columns)}")
    frame = frame.loc[frame["class_name"] == "clock"].copy()
    if frame.empty:
        raise RuntimeError("Joined CSV contains no clock rows")
    annotations = pd.read_csv(args.annotations)
    required_annotation_columns = {
        "relative_path",
        "has_digital",
        "baseline_error_type",
    }
    missing_annotation_columns = required_annotation_columns - set(annotations.columns)
    if missing_annotation_columns:
        raise ValueError(
            "Annotations are missing columns: "
            f"{sorted(missing_annotation_columns)}"
        )
    if annotations["relative_path"].duplicated().any():
        raise ValueError("Annotations contain duplicate relative_path values")
    frame = frame.merge(
        annotations, on="relative_path", how="left", validate="one_to_one"
    )
    if frame["has_digital"].isna().any():
        raise RuntimeError("Missing manual clock labels")
    frame["has_digital"] = pd.to_numeric(frame["has_digital"], errors="raise")
    if not frame["has_digital"].isin([0, 1]).all():
        raise ValueError("has_digital must contain only 0 or 1")
    frame["has_digital"] = frame["has_digital"].astype(bool)
    if set(frame["has_digital"].unique()) != {False, True}:
        raise RuntimeError("Both analog-only and analog+digital groups are required")

    condition = (
        frame.groupby("has_digital", observed=True)
        .agg(
            samples=("failure", "size"),
            exact=("failure", lambda values: 1.0 - values.mean()),
            failures=("failure", "sum"),
            under=("under", "sum"),
            over=("over", "sum"),
            mae=("abs_error", "mean"),
            bias=("signed_error", "mean"),
        )
        .reset_index()
    )
    condition["condition"] = condition["has_digital"].map(
        {False: "analog_only", True: "analog_and_digital"}
    )
    condition.to_csv(args.output_dir / "clock_condition_summary.csv", index=False)

    by_count = (
        frame.groupby(["gt_count", "has_digital"], observed=True)
        .agg(
            samples=("failure", "size"),
            exact=("failure", lambda values: 1.0 - values.mean()),
            failures=("failure", "sum"),
            under=("under", "sum"),
            over=("over", "sum"),
        )
        .reset_index()
    )
    by_count["condition"] = by_count["has_digital"].map(
        {False: "analog_only", True: "analog_and_digital"}
    )
    by_count.to_csv(args.output_dir / "clock_condition_by_count.csv", index=False)

    digital = frame["has_digital"]
    table = [
        [
            int((digital & frame["under"]).sum()),
            int((digital & ~frame["under"]).sum()),
        ],
        [
            int((~digital & frame["under"]).sum()),
            int((~digital & ~frame["under"]).sum()),
        ],
    ]
    odds_ratio, p_value = fisher_exact(table)
    result = {
        "table_rows": ["analog_and_digital", "analog_only"],
        "table_columns": ["under", "not_under"],
        "table": table,
        "odds_ratio": float(odds_ratio),
        "fisher_exact_two_sided_p": float(p_value),
        "manual_label_definition": (
            "has_digital=1 when at least one distinct clock uses a numeric "
            "digital/flip-style time display."
        ),
    }
    with (args.output_dir / "clock_condition_fisher.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)

    plot = condition.set_index("condition").loc[
        ["analog_only", "analog_and_digital"]
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Analog only", "Analog + digital"],
        1.0 - plot["exact"],
        color=["#5B9BD5", "#ED7D31"],
    )
    for index, (_, row) in enumerate(plot.iterrows()):
        ax.text(
            index,
            1.0 - row["exact"] + 0.015,
            f"{int(row['failures'])}/{int(row['samples'])}",
            ha="center",
        )
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("Failure rate at threshold 0.4")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(args.output_dir / "clock_digital_condition.png", dpi=180)
    plt.close(fig)

    print(json.dumps(result, indent=2))
    print(condition.to_string(index=False))


if __name__ == "__main__":
    main()
