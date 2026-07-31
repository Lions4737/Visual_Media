#!/usr/bin/env python3
"""Analyze the existing CountGD run on the 8,000-image GPT-Image dataset.

The directory-requested count is treated as a *pseudo ground truth*.  It is useful
for screening, but selected failure images still require manual recounting before
they are used as evidence in the report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--countgd-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "failure_analysis",
    )
    return parser.parse_args()


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    half = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
        / den
    )
    return center - half, center + half


def load_metadata(dataset_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metadata_path = dataset_root / "metadata.jsonl"
    with metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rel = Path(item["file_name"])
            count_word = rel.parts[0]
            class_plural = rel.parts[1].split("_", 1)[1]
            base = f"a photo of {count_word} {class_plural}"
            prompt = str(item["text"])
            if prompt.startswith(base):
                context = prompt[len(base) :].strip() or "plain"
            else:
                context = "unparsed"
            rows.append(
                {
                    "relative_path": rel.as_posix(),
                    "generation_prompt": prompt,
                    "generation_count": int(item["count"]),
                    "generation_context": context,
                    "count_word": count_word,
                    "class_plural": class_plural,
                }
            )
    return pd.DataFrame(rows)


def load_joined(dataset_root: Path, countgd_csv: Path) -> pd.DataFrame:
    dataset_root = dataset_root.resolve()
    frame = pd.read_csv(countgd_csv)
    required = {"image_path", "text", "count", "gt_count"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CountGD CSV is missing columns: {sorted(missing)}")
    if "confidence_threshold" in frame:
        thresholds = pd.to_numeric(frame["confidence_threshold"], errors="raise")
        if not thresholds.eq(0.4).all():
            raise ValueError("This analysis requires confidence_threshold=0.4")
    else:
        print(
            "Warning: legacy CountGD CSV has no confidence_threshold column; "
            "assuming the frozen value 0.4."
        )

    def relative_path(value: object) -> str:
        path = Path(str(value))
        if path.is_absolute():
            return path.resolve().relative_to(dataset_root).as_posix()
        return path.as_posix()

    frame["relative_path"] = frame["image_path"].map(
        relative_path
    )
    if "heatmap_path" not in frame:
        frame["heatmap_path"] = ""
    frame = frame.merge(
        load_metadata(dataset_root),
        on="relative_path",
        how="left",
        validate="one_to_one",
    )
    if frame["generation_prompt"].isna().any():
        missing_paths = frame.loc[
            frame["generation_prompt"].isna(), "relative_path"
        ].tolist()
        raise RuntimeError(
            "CountGD rows are missing from dataset metadata: "
            f"{missing_paths[:10]}"
        )
    frame = frame.rename(columns={"text": "class_name", "count": "pred_count"})
    frame["gt_count"] = frame["gt_count"].astype(int)
    frame["pred_count"] = frame["pred_count"].astype(int)
    if not frame["generation_count"].astype(int).eq(frame["gt_count"]).all():
        raise RuntimeError("Directory GT count disagrees with dataset metadata")
    frame["signed_error"] = frame["pred_count"] - frame["gt_count"]
    frame["abs_error"] = frame["signed_error"].abs()
    frame["failure"] = frame["signed_error"].ne(0)
    frame["under"] = frame["signed_error"].lt(0)
    frame["over"] = frame["signed_error"].gt(0)
    frame["severe_failure"] = frame["abs_error"].ge(2)
    return frame


def add_failure_ci(summary: pd.DataFrame) -> pd.DataFrame:
    intervals = [
        wilson_interval(int(failures), int(samples))
        for failures, samples in zip(summary["failures"], summary["samples"])
    ]
    summary["failure_ci95_low"] = [value[0] for value in intervals]
    summary["failure_ci95_high"] = [value[1] for value in intervals]
    return summary


def grouped_summary(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    summary = (
        frame.groupby(key, observed=True)
        .agg(
            samples=("failure", "size"),
            failures=("failure", "sum"),
            failure_rate=("failure", "mean"),
            exact_match=("failure", lambda values: 1.0 - values.mean()),
            mae=("abs_error", "mean"),
            bias=("signed_error", "mean"),
            under=("under", "sum"),
            over=("over", "sum"),
            severe_failures=("severe_failure", "sum"),
            max_abs_error=("abs_error", "max"),
        )
        .reset_index()
    )
    return add_failure_ci(summary)


def class_context_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["class_name", "generation_context"], observed=True)
        .agg(
            samples=("failure", "size"),
            failures=("failure", "sum"),
            failure_rate=("failure", "mean"),
            mae=("abs_error", "mean"),
            bias=("signed_error", "mean"),
        )
        .reset_index()
    )


def road_effects(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for class_name, group in frame.groupby("class_name", observed=True):
        on_road = group["generation_context"].eq("on the road")
        table = np.array(
            [
                [(on_road & group["failure"]).sum(), (on_road & ~group["failure"]).sum()],
                [
                    (~on_road & group["failure"]).sum(),
                    (~on_road & ~group["failure"]).sum(),
                ],
            ],
            dtype=int,
        )
        if table[0].sum() == 0 or table[1].sum() == 0:
            continue
        odds_ratio, p_value = fisher_exact(table)
        rows.append(
            {
                "class_name": class_name,
                "road_samples": int(table[0].sum()),
                "road_failures": int(table[0, 0]),
                "road_failure_rate": table[0, 0] / table[0].sum(),
                "other_samples": int(table[1].sum()),
                "other_failures": int(table[1, 0]),
                "other_failure_rate": table[1, 0] / table[1].sum(),
                "odds_ratio": float(odds_ratio),
                "fisher_p": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values("fisher_p") if not result.empty else result


def association_tests(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    tests: dict[str, dict[str, float | int]] = {}
    for column in ["class_name", "gt_count", "generation_context"]:
        contingency = pd.crosstab(frame[column], frame["failure"])
        result = chi2_contingency(contingency)
        tests[column] = {
            "chi2": float(result.statistic),
            "degrees_of_freedom": int(result.dof),
            "p_value": float(result.pvalue),
        }
    return tests


def save_plots(
    frame: pd.DataFrame, class_summary: pd.DataFrame, output_dir: Path
) -> None:
    sns.set_theme(style="whitegrid")

    ordered = class_summary.sort_values("failure_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    lower = ordered["failure_rate"] - ordered["failure_ci95_low"]
    upper = ordered["failure_ci95_high"] - ordered["failure_rate"]
    ax.barh(ordered["class_name"], ordered["failure_rate"], color="#4472C4")
    ax.errorbar(
        ordered["failure_rate"],
        ordered["class_name"],
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#222222",
        capsize=2,
        linewidth=0.8,
    )
    ax.set_xlabel("Failure rate against directory-requested count")
    ax.set_ylabel("")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(output_dir / "class_failure_rates.png", dpi=180)
    plt.close(fig)

    matrix = frame.pivot_table(
        index="class_name",
        columns="gt_count",
        values="failure",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0%",
        cmap="YlOrRd",
        vmin=0,
        vmax=max(0.30, float(matrix.max().max())),
        ax=ax,
    )
    ax.set_xlabel("Directory-requested count")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "class_count_failure_heatmap.png", dpi=180)
    plt.close(fig)

    errors = (
        frame.loc[frame["failure"]]
        .groupby(["class_name", "under"], observed=True)
        .size()
        .unstack(fill_value=0)
        .rename(columns={True: "under", False: "over"})
    )
    errors = errors.reindex(columns=["under", "over"], fill_value=0)
    errors = errors.reindex(
        class_summary.sort_values("failures", ascending=False)["class_name"]
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    errors[["under", "over"]].plot.bar(
        stacked=True, color=["#ED7D31", "#5B9BD5"], ax=ax
    )
    ax.set_xlabel("")
    ax.set_ylabel("Number of failed images")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(output_dir / "failure_direction_by_class.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_joined(args.dataset_root, args.countgd_csv)

    class_summary = grouped_summary(frame, "class_name").sort_values(
        ["failure_rate", "mae"], ascending=False
    )
    count_summary = grouped_summary(frame, "gt_count").sort_values("gt_count")
    context_summary = grouped_summary(frame, "generation_context").sort_values(
        "failure_rate", ascending=False
    )
    class_context = class_context_summary(frame)
    road = road_effects(frame)

    class_count = (
        frame.groupby(["class_name", "gt_count"], observed=True)
        .agg(
            samples=("failure", "size"),
            failures=("failure", "sum"),
            failure_rate=("failure", "mean"),
            mae=("abs_error", "mean"),
            bias=("signed_error", "mean"),
        )
        .reset_index()
    )

    failure_rows = frame.loc[
        frame["failure"],
        [
            "relative_path",
            "image_path",
            "heatmap_path",
            "class_name",
            "gt_count",
            "pred_count",
            "signed_error",
            "abs_error",
            "generation_context",
            "generation_prompt",
        ],
    ].sort_values(["abs_error", "class_name"], ascending=[False, True])

    frame.to_csv(args.output_dir / "joined_existing_results.csv", index=False)
    class_summary.to_csv(args.output_dir / "class_summary.csv", index=False)
    count_summary.to_csv(args.output_dir / "count_summary.csv", index=False)
    context_summary.to_csv(args.output_dir / "context_summary.csv", index=False)
    class_count.to_csv(args.output_dir / "class_count_summary.csv", index=False)
    class_context.to_csv(args.output_dir / "class_context_summary.csv", index=False)
    road.to_csv(args.output_dir / "road_vs_other_by_class.csv", index=False)
    failure_rows.to_csv(args.output_dir / "failure_rows.csv", index=False)

    overall_failures = int(frame["failure"].sum())
    class_failures = class_summary.set_index("class_name")["failures"]
    top_five = int(class_failures.nlargest(5).sum())
    report = {
        "dataset_samples": int(len(frame)),
        "pseudo_gt_warning": (
            "gt_count comes from the generation request/directory. "
            "Manually recount selected evidence images."
        ),
        "threshold": 0.4,
        "overall": {
            "failures": overall_failures,
            "failure_rate": float(frame["failure"].mean()),
            "exact_match": float(1.0 - frame["failure"].mean()),
            "mae": float(frame["abs_error"].mean()),
            "bias": float(frame["signed_error"].mean()),
            "under": int(frame["under"].sum()),
            "over": int(frame["over"].sum()),
            "severe_failures": int(frame["severe_failure"].sum()),
        },
        "failure_concentration": {
            "top_five_classes": class_failures.nlargest(5).index.tolist(),
            "top_five_failures": top_five,
            "share_of_all_failures": (
                top_five / overall_failures if overall_failures else 0.0
            ),
        },
        "association_tests": association_tests(frame),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    save_plots(frame, class_summary, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
