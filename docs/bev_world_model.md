# BEV Occupancy Forecasting and Trajectory Reranking

This engineering extension forecasts future LiDAR occupancy from recorded
history, then scores Alpamayo trajectory candidates against that forecast. The
model is intentionally compact enough for one GPU. No trained checkpoint or
world-model result is committed; the commands below define the experiment.

## Input contract

The LiDAR oracle output is consumed directly. No enrichment pass is required.
Each JSONL or CSV manifest row contains `clip_id`, `t0_us`, `chunk_id`, and
`artifact_path`; relative artifact paths resolve from the manifest directory.
`chunk_id` identifies the source LiDAR archive and is the split group.

Required pickle-free NPZ fields are:

| Key | Shape | Meaning |
| --- | ---: | --- |
| `schema_version`, `clip_id`, `t0_us` | scalar | Artifact identity checked against the manifest; schema version 4 is required |
| `past_occupancy` | `[P,H,W]` | Historical endpoint occupancy |
| `past_observed` | `[P,H,W]` | Historical ray-observed mask |
| `history_offsets_s` | `[P]` | Strictly increasing, non-positive history times |
| `occupancy` | `[F,H,W]` | Recorded-future occupancy target |
| `observed` | `[F,H,W]` | Valid target cells for loss and metrics |
| `horizons_s` | `[F]` | Strictly increasing, positive forecast times |
| `bev_x_min_m`, `bev_y_min_m` | scalar | Metric origin of grid cell `[0,0]` |
| `bev_resolution_m` | scalar | Metres per cell |
| `frame` | scalar string | Must be `ego_at_t0` |

The loader also accepts the explicit aliases `future_occupancy`,
`future_visibility`, `past_visibility`, `past_horizons_s`,
`grid_origin_xy_m`, `resolution_m`, and `coordinate_frame`. Occupancy and
observed arrays are required, binary, finite, and shape-compatible.

BEV arrays are indexed `[y,x]`: columns increase with ego-forward `x`, rows
increase with ego-left `y`.

## Leakage-safe splits

```bash
python -m src.world_model.split_manifest \
  --manifest results/world_oracle/manifests/all.jsonl \
  --output-dir results/world_model/splits \
  --seed 2026 --ratios 0.8,0.1,0.1
```

Each `chunk_id` is assigned by a stable SHA-256 hash of the seed and chunk. The
assignment is independent of manifest order, and one chunk cannot cross splits.
Training also rejects any source chunk that occurs in both the training and
validation manifests. With very few chunks, a split can be empty; inspect
`split_summary.json` before training.

## Train and resume

```bash
python -m src.world_model.train \
  --train-manifest results/world_model/splits/train.jsonl \
  --val-manifest results/world_model/splits/val.jsonl \
  --output-dir results/world_model/run_001 \
  --batch-size 4 --amp

python -m src.world_model.train \
  --train-manifest results/world_model/splits/train.jsonl \
  --val-manifest results/world_model/splits/val.jsonl \
  --output-dir results/world_model/run_001 \
  --epochs 30 --resume auto --amp
```

The model encodes each history grid, aggregates time with a convolutional GRU,
and decodes all horizons jointly. Loss is class-balanced BCE plus Dice, masked
by `observed`. Every epoch atomically writes `latest.pt`; an improved validation
loss also writes `best.pt`. A resumable checkpoint includes model, optimizer,
GradScaler and RNG states, manifest hashes, source chunk sets, a SHA-256 digest
for every oracle artifact, geometry, run configuration, and history. The source
digests form a canonical dataset fingerprint. Resume fails if an artifact changes
in place or if another invariant differs. `--epochs` may increase.

## Evaluation and prediction artifacts

Run the learned model and static persistence baseline on the same held-out
manifest:

```bash
python -m src.world_model.evaluate \
  --manifest results/world_model/splits/test.jsonl --method learned \
  --checkpoint results/world_model/run_001/best.pt \
  --output results/world_model/learned_test.json \
  --prediction-dir results/world_model/predictions --amp

python -m src.world_model.evaluate \
  --manifest results/world_model/splits/test.jsonl --method persistence \
  --output results/world_model/persistence_test.json
```

Metrics are visibility-masked IoU, precision, recall, histogram AUPRC, and Brier
score per horizon and as an unweighted horizon mean. Prediction filenames use a
SHA-256 identity digest. Each atomically written NPZ embeds clip identity,
geometry, producer method, source-artifact hash, checkpoint hash, and canonical
run fingerprint. Existing predictions are accepted only when those fields
match; use `--overwrite-predictions` for an intentional replacement. Learned
evaluation rejects a manifest that shares a source chunk with the checkpoint's
training data.

## Selection-time boundary

```bash
python -m src.world_model.rerank \
  --manifest results/world_oracle/manifests/all.jsonl \
  --prediction-dir results/world_model/predictions \
  --output results/world_model/reranked.jsonl
```

Reranking reads only predicted occupancy and deployable candidate information:
trajectory positions and times, optional yaw, vehicle dimensions, clip identity,
and BEV geometry. It does not load `occupancy`, `observed`, oracle ranks, or
`oracle_safest_idx`, even if those evaluation fields coexist in the NPZ. The
output contains no oracle choice. It records the selected index, component
scores, prediction provenance, resolved input paths, and SHA-256 hashes.

The score combines swept-footprint collision probability, predictive entropy,
out-of-bounds exposure, acceleration, jerk, curvature, and progress. Fix weights
on validation data before evaluating a held-out test set. Recorded-future risk
and ADE comparisons must be joined after selection in a separate evaluator.

## SLURM

Both scripts execute workload commands through `srun` and default to one GPU on
`dualcard`. An `sbatch --partition=...` option overrides the default.

```bash
WORLD_TRAIN_MANIFEST=results/world_model/splits/train.jsonl \
WORLD_VAL_MANIFEST=results/world_model/splits/val.jsonl \
WORLD_OUTPUT_DIR=results/world_model/run_001 \
sbatch slurm/train_world_model.sh

WORLD_EVAL_MANIFEST=results/world_model/splits/test.jsonl \
WORLD_CHECKPOINT=results/world_model/run_001/best.pt \
WORLD_METRICS_OUTPUT=results/world_model/learned_test.json \
WORLD_PREDICTION_DIR=results/world_model/predictions \
sbatch slurm/evaluate_world_model.sh
```

Set `WORLD_PYTHON_BIN` for the project interpreter and `WORLD_RESUME=auto` to
resume. The scripts fail before training if input paths or CUDA are unavailable.

## Decision gates

Scale beyond a smoke run only if:

1. 100 oracle artifacts load with no identity, geometry, or schema failure.
2. Learned IoU and AUPRC exceed persistence at short horizons and Brier is lower.
3. Candidate diversity is sufficient for selection to change outcomes.
4. On held-out clips, selection reduces recorded-future collision exposure by at
   least 15% relative to candidate zero.
5. Mean ADE degradation is at most 0.2 m, reported with paired bootstrap
   intervals.

These are acceptance thresholds, not results from the current repository.
