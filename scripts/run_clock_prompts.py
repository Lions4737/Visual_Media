#!/usr/bin/env python3
"""Run a frozen CountGD clock-prompt sweep and retain detection centers/scores.

This script deliberately lives outside the CountGD repository so the user's
existing inference code and previous results remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CONFIDENCE_THRESHOLD = 0.4


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="CSV manifest; repeat to combine frozen partitions",
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=repo_root / "configs" / "clock_prompt_candidates_one_shot.json",
    )
    parser.add_argument("--countgd-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--text-encoder-path", type=Path)
    parser.add_argument(
        "--split", choices=["discovery", "heldout", "all"], default="all"
    )
    parser.add_argument(
        "--prompt-id",
        action="append",
        dest="prompt_ids",
        help="Run only this frozen pilot prompt id; may be repeated.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Default: outputs/clock_prompt_<split>",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--resume", action="store_true")
    output_mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(paths: list[Path], split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"split", "failure_type", "gt_count", "relative_path"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Manifest {path} is missing columns: {sorted(missing)}"
                )
            rows.extend(reader)
    if split != "all":
        rows = [row for row in rows if row["split"] == split]
    seen: set[str] = set()
    for row in rows:
        rel = row["relative_path"]
        if rel in seen:
            raise ValueError(f"Duplicate image in selected manifest split: {rel}")
        seen.add(rel)
    return rows


def read_prompts(path: Path, selected: list[str] | None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    prompts = config["pilot_prompts"]
    selected = selected or ["canonical", "digital"]
    if selected:
        wanted = set(selected)
        prompts = [prompt for prompt in prompts if prompt["id"] in wanted]
        missing = wanted - {prompt["id"] for prompt in prompts}
        if missing:
            raise ValueError(f"Unknown prompt ids: {sorted(missing)}")
    return prompts


def load_existing(path: Path) -> tuple[list[dict[str, object]], set[tuple[str, str]]]:
    rows: list[dict[str, object]] = []
    completed: set[tuple[str, str]] = set()
    if not path.exists():
        return rows, completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["relative_path"]), str(row["prompt_id"]))
            if key in completed:
                raise RuntimeError(f"Duplicate row in existing output: {key}")
            rows.append(row)
            completed.add(key)
    return rows, completed


def build_model(args: argparse.Namespace):
    from run_countgd_batch import build_model_and_transforms

    model_args = Namespace(
        countgd_root=str(args.countgd_root),
        config=str(args.config) if args.config else None,
        text_encoder_path=(
            str(args.text_encoder_path) if args.text_encoder_path else None
        ),
        pretrain_model_path=str(args.checkpoint),
        device=args.device,
        seed=args.seed,
    )
    return build_model_and_transforms(model_args)


def infer_one(
    model,
    transform,
    device: torch.device,
    image_path: Path,
    prompt: str,
    min_threshold: float,
) -> tuple[list[dict[str, object]], float]:
    image = Image.open(image_path).convert("RGB")
    input_image, target = transform(image, {"exemplars": torch.tensor([])})
    input_image = input_image.to(device)
    exemplar = target["exemplars"].to(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(
            input_image.unsqueeze(0),
            [exemplar],
            [torch.tensor([0], device=device)],
            captions=[prompt + " ."],
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    scores = output["pred_logits"][0].sigmoid().max(dim=-1).values
    boxes = output["pred_boxes"][0]
    keep = scores > min_threshold
    scores = scores[keep].detach().cpu().numpy()
    boxes = boxes[keep].detach().cpu().numpy()
    order = np.argsort(-scores)
    detections = [
        {
            "score": float(scores[index]),
            "cx": float(boxes[index, 0]),
            "cy": float(boxes[index, 1]),
            "w": float(boxes[index, 2]),
            "h": float(boxes[index, 3]),
        }
        for index in order
    ]
    return detections, elapsed


def write_summary(
    raw_rows: list[dict[str, object]], thresholds: list[float], path: Path
) -> None:
    fields = [
        "split",
        "failure_type",
        "relative_path",
        "gt_count",
        "prompt_id",
        "prompt_category",
        "prompt",
        "threshold",
        "pred_count",
        "signed_error",
        "abs_error",
        "max_score",
        "inference_seconds",
        "centers_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for raw in raw_rows:
            detections = list(raw["detections"])
            for threshold in thresholds:
                selected = [
                    det for det in detections if float(det["score"]) > threshold
                ]
                pred_count = len(selected)
                gt_count = int(raw["gt_count"])
                signed_error = pred_count - gt_count
                writer.writerow(
                    {
                        "split": raw["split"],
                        "failure_type": raw["failure_type"],
                        "relative_path": raw["relative_path"],
                        "gt_count": gt_count,
                        "prompt_id": raw["prompt_id"],
                        "prompt_category": raw["prompt_category"],
                        "prompt": raw["prompt"],
                        "threshold": threshold,
                        "pred_count": pred_count,
                        "signed_error": signed_error,
                        "abs_error": abs(signed_error),
                        "max_score": raw["max_score"],
                        "inference_seconds": raw["inference_seconds"],
                        "centers_json": json.dumps(
                            [
                                [det["cx"], det["cy"], det["score"]]
                                for det in selected
                            ],
                            separators=(",", ":"),
                        ),
                    }
                )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.output_prefix is None:
        args.output_prefix = repo_root / "outputs" / f"clock_prompt_{args.split}"
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_prefix.with_suffix(".jsonl")
    summary_path = args.output_prefix.with_name(
        args.output_prefix.name + "_summary.csv"
    )

    if args.overwrite and raw_path.exists():
        raw_path.unlink()
    elif raw_path.exists() and not args.resume:
        raise FileExistsError(
            f"{raw_path} already exists. Use --resume or --overwrite explicitly."
        )

    images = read_manifest(args.manifest, args.split)
    prompts = read_prompts(args.prompt_config, args.prompt_ids)
    if not images:
        raise RuntimeError(f"No manifest rows selected for split={args.split!r}")
    if not prompts:
        raise RuntimeError("No prompts selected")
    for row in images:
        path = args.dataset_root / row["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(path)

    target_keys = {
        (image_row["relative_path"], prompt["id"])
        for image_row in images
        for prompt in prompts
    }
    image_by_path = {row["relative_path"]: row for row in images}
    prompt_by_id = {prompt["id"]: prompt for prompt in prompts}
    raw_rows, completed = load_existing(raw_path)
    foreign_keys = completed - target_keys
    if foreign_keys:
        raise RuntimeError(
            "Existing output contains rows outside this run; choose a different "
            f"--output-prefix or use --overwrite: {sorted(foreign_keys)[:5]}"
        )
    for raw in raw_rows:
        relative_path = str(raw["relative_path"])
        prompt_id = str(raw["prompt_id"])
        image_row = image_by_path[relative_path]
        prompt = prompt_by_id[prompt_id]
        expected = {
            "split": image_row["split"],
            "failure_type": image_row["failure_type"],
            "gt_count": int(image_row["gt_count"]),
            "prompt": prompt["text"],
            "prompt_category": prompt["category"],
        }
        for field, value in expected.items():
            if raw.get(field) != value:
                raise RuntimeError(
                    f"Existing output disagrees on {field} for "
                    f"{relative_path} / {prompt_id}; use --overwrite"
                )
        if float(raw.get("min_saved_threshold", 0.0)) > CONFIDENCE_THRESHOLD:
            raise RuntimeError(
                "Existing detections were truncated above the fixed threshold; "
                "use --overwrite"
            )

    thresholds = [CONFIDENCE_THRESHOLD]
    min_threshold = CONFIDENCE_THRESHOLD
    total = len(target_keys)
    done = len(completed & target_keys)
    print(
        f"CountGD on {args.device}; {len(images)} images x {len(prompts)} "
        f"prompts, {done} completed pairs."
    )
    if done < total:
        model, transform, device = build_model(args)
    with raw_path.open("a", encoding="utf-8") as raw_handle:
        for image_row in images:
            image_path = args.dataset_root / image_row["relative_path"]
            for prompt in prompts:
                key = (image_row["relative_path"], prompt["id"])
                if key in completed:
                    continue
                detections, elapsed = infer_one(
                    model,
                    transform,
                    device,
                    image_path,
                    prompt["text"],
                    min_threshold,
                )
                raw = {
                    "split": image_row["split"],
                    "failure_type": image_row["failure_type"],
                    "relative_path": image_row["relative_path"],
                    "gt_count": int(image_row["gt_count"]),
                    "prompt_id": prompt["id"],
                    "prompt_category": prompt["category"],
                    "prompt": prompt["text"],
                    "min_saved_threshold": min_threshold,
                    "max_score": max(
                        (float(det["score"]) for det in detections), default=0.0
                    ),
                    "inference_seconds": elapsed,
                    "detections": detections,
                }
                raw_handle.write(json.dumps(raw, ensure_ascii=False) + "\n")
                raw_handle.flush()
                raw_rows.append(raw)
                completed.add(key)
                done += 1
                print(
                    f"[{done:03d}/{total:03d}] {image_row['relative_path']} | "
                    f"{prompt['text']} | >{min_threshold:g}: {len(detections)} | "
                    f"{elapsed:.2f}s",
                    flush=True,
                )

    raw_rows.sort(key=lambda row: (str(row["relative_path"]), str(row["prompt_id"])))
    write_summary(raw_rows, thresholds, summary_path)
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
