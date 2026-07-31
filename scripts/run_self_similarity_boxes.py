#!/usr/bin/env python3
"""Rerun the two frozen examples and retain CountGD bounding boxes.

The original baseline CSV retained counts and heatmap paths but not box width,
height, or confidence.  This runner therefore repeats only the two frozen
image/prompt pairs with the same strict score > 0.4 rule.  It refuses to write
the artifact if either count differs from the original baseline result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .run_clock_prompts import (
        CONFIDENCE_THRESHOLD,
        build_model,
        infer_one,
    )
else:
    from run_clock_prompts import (
        CONFIDENCE_THRESHOLD,
        build_model,
        infer_one,
    )


FROZEN_SEED = 42
EXPECTED_CHECKPOINT_SHA256 = (
    "c1bab864b17db345b4c6e3aaabb5765bc2c0a90d0bc8defb5e664a74a50aa126"
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--examples",
        type=Path,
        default=repo_root / "configs" / "self_similarity_examples.json",
    )
    parser.add_argument("--countgd-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--text-encoder-path", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seed",
        type=int,
        choices=[FROZEN_SEED],
        default=FROZEN_SEED,
        help="frozen experiment seed (only 42 is accepted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "outputs" / "self_similarity_boxes.jsonl",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_examples(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    examples = config.get("examples", [])
    if len(examples) != 2:
        raise ValueError("The frozen bounding-box run requires two examples")
    required = {
        "relative_path",
        "class_name",
        "requested_count",
        "predicted_count",
    }
    for example in examples:
        missing = required - set(example)
        if missing:
            raise ValueError(f"Frozen example is missing fields: {sorted(missing)}")
    return examples


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
    examples = load_examples(args.examples)
    for example in examples:
        image_path = args.dataset_root / str(example["relative_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
    checkpoint_sha256 = file_sha256(args.checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Checkpoint SHA-256 does not match the frozen baseline: "
            f"expected {EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_sha256}"
        )

    model, transform, device = build_model(args)
    rows: list[dict[str, object]] = []
    mismatches: list[str] = []
    for index, example in enumerate(examples, start=1):
        relative_path = str(example["relative_path"])
        prompt = str(example["class_name"])
        detections, elapsed = infer_one(
            model,
            transform,
            device,
            args.dataset_root / relative_path,
            prompt,
            CONFIDENCE_THRESHOLD,
        )
        expected_count = int(example["predicted_count"])
        if len(detections) != expected_count:
            mismatches.append(
                f"{relative_path}: expected {expected_count}, got {len(detections)}"
            )
        rows.append(
            {
                "relative_path": relative_path,
                "class_name": prompt,
                "prompt": prompt,
                "requested_count": int(example["requested_count"]),
                "predicted_count": len(detections),
                "confidence_rule": f"score > {CONFIDENCE_THRESHOLD}",
                "min_saved_threshold": CONFIDENCE_THRESHOLD,
                "seed": args.seed,
                "device": str(device),
                "checkpoint_sha256": checkpoint_sha256,
                "inference_seconds": elapsed,
                "detections": detections,
            }
        )
        print(
            f"[{index}/{len(examples)}] {relative_path} | {prompt} | "
            f">{CONFIDENCE_THRESHOLD}: {len(detections)} | {elapsed:.2f}s",
            flush=True,
        )

    if mismatches:
        raise RuntimeError(
            "The rerun does not reproduce the frozen baseline count: "
            + "; ".join(mismatches)
            + ". Do not present these boxes as the original baseline result."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
