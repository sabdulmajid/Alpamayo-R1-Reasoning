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
  --protocol results/world_model/protocol.json \
  --train-manifest results/world_model/splits/train.jsonl \
  --val-manifest results/world_model/splits/val.jsonl \
  --output-dir results/world_model/run_001 \
  --batch-size 4 --amp

python -m src.world_model.train \
  --protocol results/world_model/protocol.json \
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
  --protocol results/world_model/protocol.json \
  --manifest results/world_model/splits/test.jsonl \
  --method learned --data-role test \
  --checkpoint results/world_model/run_001/best.pt \
  --selection-record results/world_model/checkpoint_selection.json \
  --evaluation-audit results/world_model/evaluation_partition.json \
  --output results/world_model/learned_test.json \
  --prediction-dir results/world_model/predictions/learned --amp

python -m src.world_model.evaluate \
  --manifest results/world_model/splits/test.jsonl --method persistence \
  --data-role test \
  --output results/world_model/persistence_test.json \
  --prediction-dir results/world_model/predictions/persistence
```

Metrics are visibility-masked IoU, precision, recall, histogram average
precision, and Brier score per horizon and as an unweighted horizon mean.
Prediction filenames use a SHA-256 identity digest. Each atomically written NPZ embeds clip identity,
geometry, producer method, source-artifact hash, checkpoint hash, and canonical
run fingerprint. Existing predictions are accepted only when those fields
match; use `--overwrite-predictions` for an intentional replacement. Learned
evaluation rejects a manifest that shares a source chunk with the checkpoint's
training data.

## Selection-time boundary

```bash
python -m src.world_model.rerank \
  --manifest results/world_model/splits/test.jsonl \
  --method learned \
  --prediction-dir results/world_model/predictions/learned \
  --protocol results/world_model/protocol.json \
  --selection-record results/world_model/checkpoint_selection.json \
  --evaluation-audit results/world_model/evaluation_partition.json \
  --output results/world_model/reranked_learned.jsonl

python -m src.world_model.rerank \
  --manifest results/world_model/splits/test.jsonl \
  --method persistence \
  --prediction-dir results/world_model/predictions/persistence \
  --output results/world_model/reranked_persistence.jsonl
```

Reranking reads only predicted occupancy and deployable candidate information:
trajectory positions and times, optional yaw, vehicle dimensions, clip identity,
and BEV geometry. It does not load `occupancy`, `observed`, oracle ranks, or
`oracle_safest_idx`, even if those evaluation fields coexist in the NPZ. The
output contains no oracle choice. It records the selected index, component
scores, prediction provenance, resolved input paths, and SHA-256 hashes.

The score samples the vehicle footprint at the six forecast horizons. It combines
an independent-horizon collision-risk proxy, predictive entropy, out-of-bounds
exposure, acceleration, jerk, curvature, and progress. The risk proxy is not a
calibrated collision probability. Fix weights on validation data before you
evaluate a held-out test set. Join recorded-future risk and ADE after selection
in a separate evaluator.

## SLURM

Both scripts execute workload commands through `srun` and default to one GPU on
`dualcard`. An `sbatch --partition=...` option overrides the default.

```bash
WORLD_TRAIN_MANIFEST=results/world_model/splits/train.jsonl \
WORLD_VAL_MANIFEST=results/world_model/splits/val.jsonl \
WORLD_PROTOCOL=results/world_model/protocol.json \
WORLD_OUTPUT_DIR=results/world_model/run_001 \
sbatch slurm/train_world_model.sh

WORLD_EVAL_MANIFEST=results/world_model/splits/test.jsonl \
WORLD_DATA_ROLE=test \
WORLD_CHECKPOINT=results/world_model/run_001/best.pt \
WORLD_PROTOCOL=results/world_model/protocol.json \
WORLD_SELECTION_RECORD=results/world_model/checkpoint_selection.json \
WORLD_EVALUATION_AUDIT=results/world_model/evaluation_partition.json \
WORLD_METRICS_OUTPUT=results/world_model/learned_test.json \
WORLD_PREDICTION_DIR=results/world_model/predictions/learned \
sbatch slurm/evaluate_world_model.sh
```

Set `WORLD_PYTHON_BIN` for the project interpreter and `WORLD_RESUME=auto` to
resume. The scripts fail before training if input paths or CUDA are unavailable.

### Full benchmark submission

`slurm/submit_world_benchmark.sh` submits the complete dependency graph. It
writes `submission.json` after SLURM accepts all jobs. The record contains the
Git commit, Python executable and version, runtime values, job resources, job
IDs, and all `afterok` dependencies.

```bash
BENCHMARK_REPO_ROOT="$(pwd)" \
BENCHMARK_PYTHON="$CONDA_PREFIX/bin/python" \
BENCHMARK_CLIP_PARQUET=/absolute/path/eval_clips_2k.parquet \
BENCHMARK_CANDIDATE_DIR=/absolute/path/candidates_k6 \
BENCHMARK_ORACLE_DIR=/absolute/path/lidar_world_oracle \
BENCHMARK_RUN_DIR=/absolute/path/world_model_run \
BENCHMARK_CANDIDATE_JOB_ID=<candidate-job-id> \
bash slurm/submit_world_benchmark.sh
```

Use `BENCHMARK_ORACLE_JOB_ID` only to reuse an active or completed oracle job.
Also set `BENCHMARK_ORACLE_SUBMISSION_RECORD` and
`BENCHMARK_ORACLE_LINEAGE_RECORD`. The submission fails before `sbatch` if these
records do not bind the same job, inputs, source, and resumed artifacts.

The production graph targets the `dualcard` partition. Each configured node has
two NVIDIA RTX A4500 GPUs, 24 CPUs, and 60,000 MB of scheduler memory. The
resource plan is:

| Stage | Concurrent jobs | GPU for each job | CPU for each job | Memory for each job |
| --- | ---: | ---: | ---: | ---: |
| LiDAR oracle | 8 array tasks | 0 | 3 | 7 GiB |
| Prepare | 1 | 0 | 4 | 24 GiB |
| Train seeds 2026 and 2027 | 2 | 1 | 4 | 28 GiB |
| Select | 1 | 0 | 2 | 12 GiB |
| Learned evaluation | 1 | 1 | 4 | 28 GiB |
| Persistence evaluation | 1 | 0 | 4 | 16 GiB |
| Rerank and compare | At most 3 | 0 | 4 | 16 GiB |
| Final aggregation | 1 | 0 | 4 | 16 GiB |

The two training jobs become eligible after the same preparation dependency.
Together, they request both GPUs, 8 CPUs, and 56 GiB. Each GPU process must see
exactly one NVIDIA RTX A4500. The stage exits if this condition is false.

## Decision gates

Scale beyond a smoke run only if:

1. 100 oracle artifacts load with no identity, geometry, or schema failure.
2. Learned IoU and average precision exceed persistence at short horizons and
   Brier score is lower.
3. Candidate diversity is sufficient for selection to change outcomes.
4. On held-out clips, selection reduces recorded-future collision exposure by at
   least 15% relative to candidate zero.
5. Mean ADE degradation is at most 0.2 m, reported with paired bootstrap
   intervals.

These are acceptance thresholds, not results from the current repository.
