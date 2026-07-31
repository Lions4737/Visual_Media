# Visual_Media: CountGD failure analysis

Source code and frozen experiment configuration for the CountGD counting-error
analysis. This repository is source-only: datasets, model checkpoints, generated
outputs, and reports are intentionally not stored here.

## Method

- CountGD is used from an external checkout pinned to commit
  `b6f362b3f5cd20db4a171faa410dfed8f2f466d8`.
- A detection is retained only when its score is strictly greater than `0.4`.
- `clock` and `digital clock` are inferred independently on every image.
- Their detections are combined with a cross-prompt connected-component union.
  Two detections are linked only when their normalized center distance is
  `<= 0.05` **and** their box IoU is `>= 0.3`. Edges are created only across
  prompts. Because the final groups are connected components, two detections
  from one prompt can still become part of the same group through a detection
  from the other prompt; this transitive behavior is pinned by a unit test.

## Setup

Use Python 3.10. Choose external locations and adjust
these example paths before running the commands:

```bash
COUNTGD_ROOT=/absolute/path/to/CountGD
CHECKPOINT_PATH=/absolute/path/to/checkpoint_fsc147_best.pth
DATASET_ROOT=/absolute/path/to/dataset_gpt-image-1.5_8000_for_lora
RUN_ROOT=/absolute/path/to/countgd_run
TEXT_ENCODER_PATH="$COUNTGD_ROOT/checkpoints/bert-base-uncased"
```

Clone and pin CountGD, then install its dependencies and this repository's
analysis dependencies. Follow the upstream CountGD instructions to build its
GroundingDINO extension and download the local BERT files.

```bash
git clone https://github.com/niki-amini-naieni/CountGD.git "$COUNTGD_ROOT"
git -C "$COUNTGD_ROOT" checkout b6f362b3f5cd20db4a171faa410dfed8f2f466d8
python -m pip install -r "$COUNTGD_ROOT/requirements.txt"
python -m pip install -r requirements.txt
```

Verify the CountGD checkpoint before use:

```bash
printf '%s  %s\n' \
  'c1bab864b17db345b4c6e3aaabb5765bc2c0a90d0bc8defb5e664a74a50aa126' \
  "$CHECKPOINT_PATH" | sha256sum --check -
```

## Run the experiment

Run the fixed-threshold baseline over the dataset. `--output_dir` remains a
required CLI argument even when heatmaps are disabled.

```bash
python scripts/run_countgd_batch.py \
  --countgd_root "$COUNTGD_ROOT" \
  --pretrain_model_path "$CHECKPOINT_PATH" \
  --text_encoder_path "$TEXT_ENCODER_PATH" \
  --input_dir "$DATASET_ROOT" \
  --output_dir "$RUN_ROOT/heatmaps" \
  --csv_path "$RUN_ROOT/countgd_th0.4.csv" \
  --text_from parent_dir \
  --gt_from parent_dir \
  --no_heatmap
```

Analyze baseline failures and the manually annotated clock condition:

```bash
python scripts/analyze_failures.py \
  --dataset-root "$DATASET_ROOT" \
  --countgd-csv "$RUN_ROOT/countgd_th0.4.csv" \
  --output-dir "$RUN_ROOT/failure_analysis"

python scripts/analyze_clock_condition.py \
  --joined-csv "$RUN_ROOT/failure_analysis/joined_existing_results.csv" \
  --annotations configs/annotations/clock_visual_annotations.csv \
  --output-dir "$RUN_ROOT/clock_condition"
```

Create the frozen evaluation manifests, then run `clock` and `digital clock`
independently over all 400 clock images:

```bash
python scripts/make_clock_manifests.py \
  --annotations configs/annotations/clock_visual_annotations.csv \
  --pilot-manifest configs/manifests/clock_prompt_manifest.csv \
  --output-dir "$RUN_ROOT/manifests"

python scripts/run_clock_prompts.py \
  --dataset-root "$DATASET_ROOT" \
  --manifest configs/manifests/clock_prompt_manifest.csv \
  --manifest "$RUN_ROOT/manifests/clock_final_eval_manifest.csv" \
  --manifest "$RUN_ROOT/manifests/clock_remaining_analog_manifest.csv" \
  --prompt-config configs/clock_prompt_candidates_one_shot.json \
  --countgd-root "$COUNTGD_ROOT" \
  --checkpoint "$CHECKPOINT_PATH" \
  --text-encoder-path "$TEXT_ENCODER_PATH" \
  --split all \
  --prompt-id canonical \
  --prompt-id digital \
  --device cpu \
  --output-prefix "$RUN_ROOT/clock_prompt_all"
```

`--resume` verifies the selected image/prompt rows, but it does not fingerprint
the model files. Use it only with the same CountGD checkout, config, checkpoint,
text encoder, seed, and device as the interrupted run; otherwise choose a new
output prefix or use `--overwrite`.

Finally, evaluate the fixed cross-prompt union. The committed manual-audit JSON
is keyed by dataset-relative image path and can be replaced with
`--manual-audits` when a separate frozen audit is needed.

```bash
python scripts/analyze_prompt_union.py \
  --raw "$RUN_ROOT/clock_prompt_all.jsonl" \
  --annotations configs/annotations/clock_visual_annotations.csv \
  --manual-audits configs/manual_audits.json \
  --output-dir "$RUN_ROOT/prompt_union" \
  --expected-images 400
```

This analysis also renders `clock_digital_condition.png` from the paired
`clock` run, with ACC shown separately for analog-only and digital-mixed images.

Render the four frozen failure examples directly from the original pixels and
saved boxes. The two output JPEGs are generated artifacts and remain ignored:

```bash
python scripts/make_failure_example_figures.py \
  --dataset-root "$DATASET_ROOT" \
  --raw "$RUN_ROOT/clock_prompt_all.jsonl" \
  --output-dir "$RUN_ROOT/failure_examples"
```

The overlays use red for `clock`, blue for `digital clock`, green for union
components, and yellow for components detected only by `digital clock`.

The original baseline CSV did not retain bounding-box coordinates. Rerun only
the two frozen same-object double-counting examples with the same strict
`score > 0.4` rule and save their boxes. The runner aborts if either rerun count
differs from the frozen baseline count, the seed is not 42, or the checkpoint
SHA-256 differs from the frozen checkpoint:

```bash
python scripts/run_self_similarity_boxes.py \
  --dataset-root "$DATASET_ROOT" \
  --countgd-root "$COUNTGD_ROOT" \
  --checkpoint "$CHECKPOINT_PATH" \
  --text-encoder-path "$TEXT_ENCODER_PATH" \
  --output "$RUN_ROOT/self_similarity_boxes.jsonl"
```

Render the original/bounding-box comparison figure:

```bash
python scripts/make_self_similarity_figure.py \
  --dataset-root "$DATASET_ROOT" \
  --detections "$RUN_ROOT/self_similarity_boxes.jsonl" \
  --output-dir "$RUN_ROOT/self_similarity_examples"
```

All paths under `RUN_ROOT` are generated artifacts and must remain outside this
repository (or under an ignored `outputs/` directory).

Run the deterministic unit tests without loading CountGD:

```bash
python -m unittest discover -s tests -v
```

CountGD remains an external MIT-licensed dependency; see its upstream
repository for installation details and license terms.
