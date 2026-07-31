#!/usr/bin/env python3
"""Run text-only CountGD inference over a directory tree.

CountGD itself, its checkpoint, the image dataset, and generated outputs remain
external to this repository. Paths written to the CSV are relative to the input
or heatmap root so the result is portable.
"""

import argparse
import csv
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import inflect


CONFIDENCE_THRESHOLD = 0.4

COUNT_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
}
WORD_TO_NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

p = inflect.engine()


def get_args_parser():
    parser = argparse.ArgumentParser("CountGD batch inference", add_help=False)
    parser.add_argument(
        "--countgd_root",
        required=True,
        help="path to a separate checkout of the CountGD repository",
    )
    parser.add_argument("--device", default="cuda", help="device to use for inference")
    parser.add_argument(
        "--pretrain_model_path",
        required=True,
        help="path to CountGD pretrained checkpoint",
    )
    parser.add_argument(
        "--config",
        help="CountGD config; default: <countgd_root>/config/cfg_fsc147_vit_b.py",
    )
    parser.add_argument(
        "--text_encoder_path",
        help=(
            "local BERT directory; default: "
            "<countgd_root>/checkpoints/bert-base-uncased when present"
        ),
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="root directory containing images",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="directory to write heatmaps",
    )
    parser.add_argument(
        "--csv_path",
        required=True,
        help="output CSV path with per-image counts",
    )
    parser.add_argument(
        "--no_heatmap",
        action="store_true",
        help="skip saving heatmap images",
    )
    parser.add_argument(
        "--text_from",
        default="parent_dir",
        choices=["parent_dir", "filename"],
        help="how to derive text prompt",
    )
    parser.add_argument(
        "--gt_from",
        default="none",
        choices=["none", "filename_prefix", "parent_dir"],
        help="how to derive GT count",
    )
    parser.add_argument(
        "--extensions",
        default="jpg,jpeg,png,webp,bmp,tif,tiff",
        help="comma-separated list of image extensions",
    )
    parser.add_argument("--seed", default=42, type=int)
    return parser


def build_model_and_transforms(args):
    countgd_root = Path(args.countgd_root).expanduser().resolve()
    if not countgd_root.is_dir():
        raise FileNotFoundError(f"CountGD repository not found: {countgd_root}")
    sys.path.insert(0, str(countgd_root))

    import datasets_inference.transforms as T
    from util.slconfig import SLConfig

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    data_transform = T.Compose([T.RandomResize([800], max_size=1333), normalize])

    config = (
        Path(args.config).expanduser().resolve()
        if args.config
        else countgd_root / "config" / "cfg_fsc147_vit_b.py"
    )
    if not config.is_file():
        raise FileNotFoundError(f"CountGD config not found: {config}")
    cfg = SLConfig.fromfile(str(config))
    text_encoder_path = (
        Path(args.text_encoder_path).expanduser().resolve()
        if args.text_encoder_path
        else countgd_root / "checkpoints" / "bert-base-uncased"
    )
    if args.text_encoder_path and not text_encoder_path.is_dir():
        raise FileNotFoundError(f"Text encoder directory not found: {text_encoder_path}")
    if text_encoder_path.exists():
        cfg.merge_from_dict({"text_encoder_type": str(text_encoder_path)})
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    from models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, _, _ = build_func(args)
    model.to(device)

    checkpoint_path = Path(args.pretrain_model_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CountGD checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")["model"]
    incompatible = model.load_state_dict(checkpoint, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint keys: {incompatible.unexpected_keys[:20]}"
        )
    if incompatible.missing_keys:
        print(
            "Warning: checkpoint has missing model keys: "
            f"{incompatible.missing_keys[:20]}"
        )
    model.eval()
    return model, data_transform, device


def text_from_parent_dir(image_path):
    parent = os.path.basename(os.path.dirname(image_path))
    text = parent.replace("_", " ").strip()
    parts = text.split()
    if parts:
        first = parts[0].lower()
        if first in COUNT_WORDS:
            parts = parts[1:]
        elif first.isdigit():
            parts = parts[1:]
    text = " ".join(parts).strip()
    return text if text else parent


def list_images(input_dir, extensions):
    exts = [e.strip().lower() for e in extensions.split(",") if e.strip()]
    patterns = [f"**/*.{ext}" for ext in exts]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(input_dir, pat), recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    return files


def parse_from_filename(image_path):
    base = os.path.basename(image_path)
    stem, _ext = os.path.splitext(base)
    parts = stem.split("__")
    gt_count = None
    text = ""
    if parts:
        try:
            gt_count = int(parts[0])
        except ValueError:
            gt_count = None
    if len(parts) > 1:
        text = parts[1].strip()
    return text, gt_count


def parse_count_token(token):
    token_norm = token.replace("_", " ").replace("-", " ").strip().lower()
    if not token_norm:
        return None
    first = token_norm.split()[0]
    if first.isdigit():
        return int(first)
    return WORD_TO_NUM.get(first)


def parse_gt_from_parent_dir(image_path):
    parent = os.path.basename(os.path.dirname(image_path))
    return parse_count_token(parent)


def singularize_text(text):
    parts = [p for p in text.split() if p]
    singular_parts = []
    for part in parts:
        singular = p.singular_noun(part)
        singular_parts.append(singular if singular else part)
    return " ".join(singular_parts)


def save_heatmap(image_path, boxes, output_path):
    (w, h) = Image.open(image_path).size
    det_map = np.zeros((h, w))
    if boxes.shape[0] > 0:
        det_map[
            (h * boxes[:, 1]).cpu().numpy().astype(int),
            (w * boxes[:, 0]).cpu().numpy().astype(int),
        ] = 1
    det_map = ndimage.gaussian_filter(det_map, sigma=(5, 5), order=0)
    plt.figure(figsize=(8, 8))
    plt.imshow(Image.open(image_path))
    plt.imshow(det_map[None, :].transpose(1, 2, 0), "jet", interpolation="none", alpha=0.7)
    plt.axis("off")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser("CountGD batch inference", parents=[get_args_parser()])
    args = parser.parse_args()

    model, transform, device = build_model_and_transforms(args)
    if not args.no_heatmap:
        os.makedirs(args.output_dir, exist_ok=True)

    images = list_images(args.input_dir, args.extensions)
    if not images:
        raise SystemExit("No images found under input_dir.")

    csv_path = Path(args.csv_path).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_path",
                "text",
                "count",
                "gt_count",
                "confidence_threshold",
                "heatmap_path",
            ]
        )

        for image_path in images:
            filename_text, filename_gt = parse_from_filename(image_path)
            if args.text_from == "parent_dir":
                text = text_from_parent_dir(image_path)
            else:
                text = filename_text
            text = singularize_text(text)

            gt_count: int | str = ""
            if args.gt_from == "filename_prefix":
                gt_count = filename_gt if filename_gt is not None else ""
            if args.gt_from == "parent_dir":
                parsed_gt = parse_gt_from_parent_dir(image_path)
                gt_count = parsed_gt if parsed_gt is not None else ""

            try:
                with Image.open(image_path) as source_image:
                    image = source_image.convert("RGB")
                input_image, target = transform(
                    image, {"exemplars": torch.tensor([])}
                )
            except OSError as exc:
                print(f"Skipping unreadable image: {image_path} ({exc})")
                continue
            input_image = input_image.to(device)
            input_exemplar = target["exemplars"].to(device)

            with torch.inference_mode():
                model_output = model(
                    input_image.unsqueeze(0),
                    [input_exemplar],
                    [torch.tensor([0]).to(device)],
                    captions=[text + " ."],
                )
            logits = model_output["pred_logits"][0].sigmoid()
            boxes = model_output["pred_boxes"][0]
            box_mask = logits.max(dim=-1).values > CONFIDENCE_THRESHOLD
            boxes = boxes[box_mask, :]
            pred_count = boxes.shape[0]

            rel_path = os.path.relpath(image_path, args.input_dir)
            heatmap_path = ""
            if not args.no_heatmap:
                heatmap_path = os.path.join(args.output_dir, rel_path)
                heatmap_path = os.path.splitext(heatmap_path)[0] + "_heatmap.jpg"
                save_heatmap(image_path, boxes, heatmap_path)

            portable_heatmap = (
                os.path.relpath(heatmap_path, args.output_dir) if heatmap_path else ""
            )
            writer.writerow(
                [
                    rel_path,
                    text,
                    pred_count,
                    gt_count,
                    CONFIDENCE_THRESHOLD,
                    portable_heatmap,
                ]
            )

    print(f"Wrote results to {args.csv_path}")


if __name__ == "__main__":
    main()
