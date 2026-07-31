#!/usr/bin/env python3
"""Evaluate a fixed-threshold spatial union of independent clock prompts.

Each prompt is run independently at confidence threshold 0.4. Detections from
different prompt runs are treated as the same object when their normalized
center distance is <= 0.05 and their boxes overlap with IoU >= 0.3. Connected
components form the final object-level union.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import binomtest, wilcoxon


PROMPTS = {"canonical": "clock", "digital": "digital clock"}
CONFIDENCE_THRESHOLD = 0.4
CENTER_DISTANCE_THRESHOLD = 0.05
IOU_THRESHOLD = 0.3


def save_clock_condition_accuracy(
    summary: pd.DataFrame, output_path: Path
) -> None:
    """Plot paired clock-prompt baseline ACC for the two visual conditions."""
    plot = summary.loc[
        summary["group_type"].eq("condition")
        & summary["method"].eq(PROMPTS["canonical"])
        & summary["confidence_threshold"].eq(CONFIDENCE_THRESHOLD)
    ].copy()
    if set(plot["group_value"]) != {"analog_only", "analog_and_digital"}:
        raise RuntimeError("Expected both frozen clock visual conditions")
    plot = plot.set_index("group_value").loc[
        ["analog_only", "analog_and_digital"]
    ]

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.bar(
        ["Analog only", "Analog + digital"],
        plot["exact_rate"],
        color=["#5B9BD5", "#ED7D31"],
        width=0.64,
        zorder=3,
    )
    for index, (_, row) in enumerate(plot.iterrows()):
        ax.text(
            index,
            float(row["exact_rate"]) + 0.025,
            (
                f"{int(row['exact_n'])}/{int(row['samples'])}\n"
                f"({float(row['exact_rate']):.2%})"
            ),
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("ACC", fontsize=16)
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        action="append",
        required=True,
        help="prompt-detection JSONL; repeat when runs are split across files",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=repo_root
        / "configs"
        / "annotations"
        / "clock_visual_annotations.csv",
    )
    parser.add_argument(
        "--manual-audits",
        type=Path,
        default=repo_root / "configs" / "manual_audits.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "prompt_union",
    )
    parser.add_argument("--expected-images", type=int, default=400)
    return parser.parse_args()


def load_rows(
    paths: list[Path], expected_images: int
) -> dict[tuple[str, str], dict[str, object]]:
    rows: dict[tuple[str, str], dict[str, object]] = {}
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
                if float(row.get("min_saved_threshold", 0.0)) > CONFIDENCE_THRESHOLD:
                    raise RuntimeError(
                        "Raw detections were truncated above the fixed analysis "
                        f"threshold for {row.get('relative_path')} / {prompt_id}"
                    )
                key = (str(row["relative_path"]), prompt_id)
                if key in rows:
                    raise RuntimeError(f"Duplicate raw prompt row: {key}")
                rows[key] = row
    images = {key[0] for key in rows}
    expected_rows = expected_images * len(PROMPTS)
    if len(images) != expected_images or len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_images} images x {len(PROMPTS)} prompts; "
            f"got {len(images)} images and {len(rows)} rows"
        )
    return rows


def box_iou(a: dict[str, object], b: dict[str, object]) -> float:
    ax1 = float(a["cx"]) - float(a["w"]) / 2
    ay1 = float(a["cy"]) - float(a["h"]) / 2
    ax2 = float(a["cx"]) + float(a["w"]) / 2
    ay2 = float(a["cy"]) + float(a["h"]) / 2
    bx1 = float(b["cx"]) - float(b["w"]) / 2
    by1 = float(b["cy"]) - float(b["h"]) / 2
    bx2 = float(b["cx"]) + float(b["w"]) / 2
    by2 = float(b["cy"]) + float(b["h"]) / 2
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (
        float(a["w"]) * float(a["h"])
        + float(b["w"]) * float(b["h"])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def selected(row: dict[str, object]) -> list[dict[str, object]]:
    return [
        dict(item)
        for item in row["detections"]
        if float(item["score"]) > CONFIDENCE_THRESHOLD
    ]


def spatial_union(
    prompt_detections: dict[str, list[dict[str, object]]],
    iou_threshold: float,
    center_distance_threshold: float = CENTER_DISTANCE_THRESHOLD,
) -> list[dict[str, object]]:
    """Return connected components linked only across different prompts."""
    detections: list[dict[str, object]] = []
    for prompt_id, items in prompt_detections.items():
        detections.extend(
            dict(item, prompt_id=prompt_id, prompt=PROMPTS[prompt_id])
            for item in items
        )

    parent = list(range(len(detections)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, left_item in enumerate(detections):
        for right in range(left + 1, len(detections)):
            right_item = detections[right]
            # The operation is a prompt-level set union. We only remove
            # duplicates supported by overlap across independent prompt runs.
            if left_item["prompt_id"] == right_item["prompt_id"]:
                continue
            center_distance = math.hypot(
                float(left_item["cx"]) - float(right_item["cx"]),
                float(left_item["cy"]) - float(right_item["cy"]),
            )
            if (
                center_distance <= center_distance_threshold
                and box_iou(left_item, right_item) >= iou_threshold
            ):
                union(left, right)

    components: dict[int, list[dict[str, object]]] = defaultdict(list)
    for index, item in enumerate(detections):
        components[find(index)].append(item)

    merged: list[dict[str, object]] = []
    for members in components.values():
        representative = max(members, key=lambda item: float(item["score"]))
        merged.append(
            {
                "cx": float(representative["cx"]),
                "cy": float(representative["cy"]),
                "w": float(representative["w"]),
                "h": float(representative["h"]),
                "score": float(representative["score"]),
                "prompts": sorted({str(item["prompt"]) for item in members}),
                "member_count": len(members),
                "members": members,
            }
        )
    return sorted(merged, key=lambda item: float(item["score"]), reverse=True)


def aggregate(
    frame: pd.DataFrame, group_type: str, group_value: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, column in [
        ("clock", "clock_count"),
        ("digital clock", "digital_count"),
        ("prompt union", "union_count"),
    ]:
        error = frame[column] - frame["requested_count"]
        rows.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "method": method,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "center_distance_threshold": (
                    CENTER_DISTANCE_THRESHOLD if method == "prompt union" else None
                ),
                "iou_threshold": IOU_THRESHOLD if method == "prompt union" else None,
                "samples": len(frame),
                "exact_n": int(error.eq(0).sum()),
                "exact_rate": float(error.eq(0).mean()),
                "mae": float(error.abs().mean()),
                "bias": float(error.mean()),
                "under": int(error.lt(0).sum()),
                "over": int(error.gt(0).sum()),
            }
        )
    return rows


def paired(
    frame: pd.DataFrame, baseline_column: str, candidate_column: str
) -> dict[str, object]:
    baseline_error = (frame[baseline_column] - frame["requested_count"]).abs()
    candidate_error = (frame[candidate_column] - frame["requested_count"]).abs()
    baseline_exact = baseline_error.eq(0)
    candidate_exact = candidate_error.eq(0)
    candidate_only = int((~baseline_exact & candidate_exact).sum())
    baseline_only = int((baseline_exact & ~candidate_exact).sum())
    discordant = candidate_only + baseline_only
    mcnemar_p = (
        float(binomtest(candidate_only, discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    differences = baseline_error - candidate_error
    if differences.eq(0).all():
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(
            wilcoxon(baseline_error, candidate_error, zero_method="pratt").pvalue
        )
        if not math.isfinite(wilcoxon_p):
            wilcoxon_p = 1.0
    return {
        "baseline": baseline_column.removesuffix("_count"),
        "candidate": candidate_column.removesuffix("_count"),
        "samples": len(frame),
        "baseline_exact": float(baseline_exact.mean()),
        "candidate_exact": float(candidate_exact.mean()),
        "candidate_only_correct": candidate_only,
        "baseline_only_correct": baseline_only,
        "candidate_lower_abs_error": int(differences.gt(0).sum()),
        "equal_abs_error": int(differences.eq(0).sum()),
        "candidate_higher_abs_error": int(differences.lt(0).sum()),
        "mcnemar_exact_p": mcnemar_p,
        "wilcoxon_abs_error_p": wilcoxon_p,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_rows(args.raw, args.expected_images)
    annotations = pd.read_csv(args.annotations)
    required_annotation_columns = {"relative_path", "has_digital"}
    missing_columns = required_annotation_columns - set(annotations.columns)
    if missing_columns:
        raise ValueError(
            f"Annotations are missing columns: {sorted(missing_columns)}"
        )
    if annotations["relative_path"].duplicated().any():
        duplicates = annotations.loc[
            annotations["relative_path"].duplicated(keep=False), "relative_path"
        ].tolist()
        raise ValueError(f"Duplicate annotation paths: {duplicates[:10]}")
    annotations["has_digital"] = pd.to_numeric(
        annotations["has_digital"], errors="raise"
    )
    if not annotations["has_digital"].isin([0, 1]).all():
        raise ValueError("has_digital must contain only 0 or 1")
    with args.manual_audits.open(encoding="utf-8") as handle:
        manual_audits = json.load(handle)
    annotation_map = annotations.set_index("relative_path")["has_digital"].to_dict()
    raw_paths = {key[0] for key in raw}
    missing_annotations = raw_paths - set(annotation_map)
    if missing_annotations:
        raise RuntimeError(
            "Missing clock annotations for: "
            f"{sorted(missing_annotations)[:10]}"
        )
    unknown_audits = set(manual_audits) - set(annotation_map)
    if unknown_audits:
        raise RuntimeError(
            f"Manual audits reference unknown images: {sorted(unknown_audits)[:10]}"
        )

    per_image: list[dict[str, object]] = []
    for relative_path in sorted({key[0] for key in raw}):
        clock_row = raw[(relative_path, "canonical")]
        digital_row = raw[(relative_path, "digital")]
        for field in ["gt_count", "split", "failure_type"]:
            if clock_row[field] != digital_row[field]:
                raise RuntimeError(
                    f"Prompt rows disagree on {field} for {relative_path}"
                )
        clock = selected(clock_row)
        digital = selected(digital_row)
        merged = spatial_union(
            {"canonical": clock, "digital": digital}, IOU_THRESHOLD
        )
        requested_count = int(clock_row["gt_count"])
        audit = manual_audits.get(relative_path, {})
        per_image.append(
            {
                "relative_path": relative_path,
                "split": str(clock_row["split"]),
                "failure_type": str(clock_row["failure_type"]),
                "condition": (
                    "analog_and_digital"
                    if bool(annotation_map[relative_path])
                    else "analog_only"
                ),
                "requested_count": requested_count,
                "clock_count": len(clock),
                "digital_count": len(digital),
                "union_count": len(merged),
                "union_signed_error_vs_request": len(merged) - requested_count,
                "visual_count_if_audited": audit.get("visual_count"),
                "audit_category": audit.get("audit_category", ""),
                "audit_note": audit.get("audit_note", ""),
                "union_components_json": json.dumps(
                    merged, ensure_ascii=False, separators=(",", ":")
                ),
            }
        )
    frame = pd.DataFrame(per_image)
    frame.to_csv(args.output_dir / "clock_prompt_union_per_image.csv", index=False)

    summary: list[dict[str, object]] = []
    summary.extend(aggregate(frame, "overall", "all"))
    for condition, subset in frame.groupby("condition", observed=True):
        summary.extend(aggregate(subset, "condition", str(condition)))
    for split, subset in frame.groupby("split", observed=True):
        summary.extend(aggregate(subset, "split", str(split)))
    development = frame.loc[frame["split"].isin(["discovery", "heldout"])]
    evaluation = frame.loc[~frame["split"].isin(["discovery", "heldout"])]
    summary.extend(
        aggregate(development, "partition", f"development_{len(development)}")
    )
    summary.extend(
        aggregate(evaluation, "partition", f"evaluation_{len(evaluation)}")
    )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(
        args.output_dir / "clock_prompt_union_summary.csv", index=False
    )
    save_clock_condition_accuracy(
        summary_frame, args.output_dir / "clock_digital_condition.png"
    )

    component_sources: list[dict[str, object]] = []
    component_groups = [("overall", "all", frame)]
    component_groups.extend(
        ("condition", str(condition), subset)
        for condition, subset in frame.groupby("condition", observed=True)
    )
    for group_type, group_value, subset in component_groups:
        counts = {"both_prompts": 0, "clock_only": 0, "digital_clock_only": 0}
        for encoded in subset["union_components_json"]:
            for component in json.loads(encoded):
                prompts = set(component["prompts"])
                if prompts == {"clock", "digital clock"}:
                    counts["both_prompts"] += 1
                elif prompts == {"clock"}:
                    counts["clock_only"] += 1
                elif prompts == {"digital clock"}:
                    counts["digital_clock_only"] += 1
                else:
                    raise RuntimeError(f"Unexpected component prompts: {prompts}")
        for source, count in counts.items():
            component_sources.append(
                {
                    "group_type": group_type,
                    "group_value": group_value,
                    "component_source": source,
                    "components": count,
                }
            )
    pd.DataFrame(component_sources).to_csv(
        args.output_dir / "clock_prompt_union_component_sources.csv", index=False
    )

    sensitivity: list[dict[str, object]] = []
    for center_threshold in [0.03, 0.04, 0.05, 0.06]:
        for iou_threshold in [0.2, 0.3, 0.4]:
            errors: list[int] = []
            for relative_path in frame["relative_path"]:
                clock = selected(raw[(relative_path, "canonical")])
                digital = selected(raw[(relative_path, "digital")])
                count = len(
                    spatial_union(
                        {"canonical": clock, "digital": digital},
                        iou_threshold,
                        center_threshold,
                    )
                )
                requested_count = int(
                    frame.loc[
                        frame["relative_path"] == relative_path, "requested_count"
                    ].iloc[0]
                )
                errors.append(count - requested_count)
            sensitivity.append(
                {
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "center_distance_threshold": center_threshold,
                    "iou_threshold": iou_threshold,
                    "samples": len(errors),
                    "exact_n": sum(error == 0 for error in errors),
                    "exact_rate": sum(error == 0 for error in errors) / len(errors),
                    "mae": sum(abs(error) for error in errors) / len(errors),
                    "under": sum(error < 0 for error in errors),
                    "over": sum(error > 0 for error in errors),
                }
            )
    pd.DataFrame(sensitivity).to_csv(
        args.output_dir / "clock_prompt_union_spatial_sensitivity.csv", index=False
    )

    request_disagreements = frame.loc[
        frame["union_signed_error_vs_request"].ne(0)
    ].copy()
    request_disagreements.to_csv(
        args.output_dir / "clock_prompt_union_request_disagreements.csv", index=False
    )

    comparisons = {
        "fixed_settings": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "center_distance_threshold": CENTER_DISTANCE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "prompts": list(PROMPTS.values()),
            "deduplication": (
                "Connected components of detections linked across different "
                "prompts when normalized center distance <= "
                "center_distance_threshold and box IoU >= iou_threshold."
            ),
        },
        "request_count_comparisons": {
            "union_vs_clock": paired(frame, "clock_count", "union_count"),
            "union_vs_digital": paired(frame, "digital_count", "union_count"),
        },
        "component_source_counts": component_sources,
        "manual_audit_of_union_request_disagreements": (
            request_disagreements[
                [
                    "relative_path",
                    "requested_count",
                    "union_count",
                    "visual_count_if_audited",
                    "audit_category",
                    "audit_note",
                ]
            ].to_dict(orient="records")
        ),
        "caveat": (
            "Request-count agreement is pseudo-GT. Manual values are included "
            "only for paths present in the manual-audits configuration."
        ),
    }
    with (args.output_dir / "clock_prompt_union_pairwise.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            comparisons,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(
        summary_frame
        .loc[lambda table: table["group_type"].isin(["overall", "condition"])]
        .to_string(index=False)
    )
    print(pd.DataFrame(sensitivity).to_string(index=False))
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
