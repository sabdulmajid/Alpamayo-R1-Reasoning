# Alpamayo-R1 Reasoning and Action Study

## Purpose

This project measures agreement between the language output and the trajectory output of Alpamayo-R1.

The model gives two outputs for each driving scene:

- A Chain of Causation (CoC) explanation
- A 6.4-second trajectory with 64 waypoints

The study compares the driving intent in the CoC with the action in the trajectory. The term `mismatch` identifies a difference between these outputs.

## Test Method

The test uses this sequence:

1. Load the camera data and the ego-motion history.
2. Generate the CoC output.
3. Generate the future trajectory.
4. Classify the intent in the CoC.
5. Classify the action in the trajectory.
6. Compare the two classes.

For example, a CoC can give an instruction to decrease speed. A trajectory that increases speed has a high mismatch score.

### Test Configuration

| Item | Value |
| --- | --- |
| Model | Alpamayo-R1-10B |
| Precision | Native `bfloat16` |
| GPU | One NVIDIA RTX A4500 with 20 GB of memory |
| Dataset | PhysicalAI Autonomous Vehicles test split |
| Evaluation size | 2,000 clips |
| Sample design | 400 clips from each of five time groups |
| Execution | Resumable batches on one GPU |

The clip sampler uses a camera-safe planning timestamp. This limit prevents invalid samples near the limits of the camera data.

## Mismatch Score

The test measures longitudinal intent and lateral intent separately.

### Longitudinal Classes

The CoC parser uses these intent classes:

- `stop`
- `slow_down`
- `maintain`
- `accelerate`

The trajectory classifier uses these action classes:

- `stopped`
- `decelerating`
- `constant_speed`
- `accelerating`

### Lateral Classes

The CoC parser uses these intent classes:

- `hold_lane`
- `nudge_left`
- `nudge_right`
- `lane_change_left`
- `lane_change_right`

The trajectory classifier uses these action classes:

- `straight`
- `shift_left`
- `shift_right`
- `lane_change_left`
- `lane_change_right`

Each axis has a documented compatibility matrix. The mismatch score is `1 - compatibility`.

A score of `0` shows full agreement. A score of `1` shows a contradiction.

## Results

The final run processed all 2,000 clips. The run had no runtime failures.

CI means 95 percent confidence interval.

| Metric | Result |
| --- | ---: |
| Valid clips | 2,000 of 2,000 |
| Mean mismatch | 0.4991, CI 0.4839 to 0.5143 |
| Mismatch standard deviation | 0.3463 |
| Mean average displacement error (ADE) | 1.9391 m, CI 1.8556 m to 2.0226 m |
| ADE standard deviation | 1.9047 m |
| Mean longitudinal match | 0.6125, CI 0.5963 to 0.6288 |
| Mean lateral match | 0.3776, CI 0.3639 to 0.3913 |

### Mismatch Levels

| Level | Score range | Clips | Rate | CI |
| --- | --- | ---: | ---: | ---: |
| Consistent | Less than 0.3 | 531 | 26.55 percent | 24.61 to 28.49 percent |
| Partial | 0.3 to less than 0.6 | 636 | 31.80 percent | 29.76 to 33.84 percent |
| Severe | 0.6 or more | 833 | 41.65 percent | 39.49 to 43.81 percent |

### Difference Between the Two Axes

The mean difference between the longitudinal match and the lateral match is 0.2349. The CI is 0.2131 to 0.2568.

The interval does not include zero. Thus, the two axes have different agreement rates in this test.

### Results for Each Time Group

| Time group | Severe mismatch rate |
| --- | ---: |
| Midday | 38.50 percent |
| Morning | 41.50 percent |
| Afternoon | 42.25 percent |
| Night | 42.75 percent |
| Evening | 43.25 percent |

Each time group has 400 clips. The difference between time groups is smaller than the total severe mismatch rate.

### Frequent Class Pairs

The most frequent longitudinal pairs are:

| CoC intent | Trajectory action | Count |
| --- | --- | ---: |
| `maintain` | `constant_speed` | 540 |
| `maintain` | `accelerating` | 316 |
| `slow_down` | `decelerating` | 204 |
| `slow_down` | `constant_speed` | 171 |

The most frequent lateral pairs are:

| CoC intent | Trajectory action | Count |
| --- | --- | ---: |
| `hold_lane` | `lane_change_right` | 198 |
| `hold_lane` | `lane_change_left` | 175 |
| `hold_lane` | `shift_right` | 160 |
| `hold_lane` | `shift_left` | 157 |
| `hold_lane` | `straight` | 134 |

## Conclusions

The test gives these conclusions:

1. The mean mismatch score is 0.4991 under this test protocol.
2. Severe mismatch occurs in 41.65 percent of the clips.
3. Longitudinal agreement is higher than lateral agreement.
4. The lateral score can include an error from the coordinate representation.

The fourth conclusion is an important limit. A curved road can cause lateral movement in the ego frame without a lane change.

Thus, the current lateral mismatch is an upper estimate. The study does not separate all representation errors from model behavior.

## Limits

The results have these limits:

- The intent classes and the compatibility matrix define the mismatch score.
- The CoC parser can assign an incorrect intent when the text is not clear.
- The lateral classifier uses movement in the ego frame.
- The test uses one model family and one dataset family.
- The test uses one GPU type and one software environment.

The correlation between mismatch and ADE is 0.1234. Thus, mismatch is not a sufficient measure of trajectory quality.

## Recommended Follow-up Tests

### Curvature Correction

Calculate lateral movement relative to a road-following baseline. Then compare the corrected result with the current result.

Use these decision limits:

- A decrease of 10 percentage points or more shows a large representation effect.
- A decrease of less than 5 percentage points shows a small representation effect.

### Parser Test

Apply strict, current, and expanded parsers to the same 2,000 trajectories.

Use these decision limits:

- A difference of more than 5 percentage points shows high parser sensitivity.
- A difference of 2 percentage points or less shows low parser sensitivity.

### Outcome Test

Compare mismatch with ADE after controls for the time group and the speed range.

A persistent ADE difference gives evidence that mismatch relates to trajectory quality. A removed difference gives evidence of classification noise.

### Human Review

Review 200 clips. Use 100 severe `hold_lane` cases and 100 matched control cases.

The review must identify a true contradiction or a representation error for each case. This review can supply labels for metric correction.

## Reproducible Candidate Generation

The file `src/generate_candidates.py` supplies a separate candidate-generation process. This process does not change the completed mismatch experiment.

The default process uses these generation values:

| Item | Value |
| --- | ---: |
| Candidate count | 6 |
| Samples in each rollout | 1 |
| Top-p | 0.98 |
| Temperature | 0.6 |
| Maximum generation length | 256 tokens |

The process pins the model, the dataset, and the Alpamayo source to immutable revisions.

The process calculates a seed from the base seed, clip ID, and planning timestamp. It also calculates and records one seed for each candidate.

The process uses six separate rollouts to make six candidates. This method avoids the device error that occurred in a two-GPU rollout test.

The validated cluster test used one NVIDIA RTX A4500 and CPU offload. The test made six different trajectories for one clip.

Exact numeric results can change with different CUDA hardware or software versions.

### Candidate Output

The process writes one set of files for each completed clip:

| Path | Content |
| --- | --- |
| `artifacts/<key>.npz` | Trajectories, sample times, seeds, text, ground truth, and clip identity |
| `records/<key>.json` | Configuration, metrics, seeds, and the artifact path |
| `manifests/shard-xxxxx-of-yyyyy.jsonl` | Completed records for one shard |

The process writes each file atomically. Each SLURM task writes a separate manifest.

The field `oracle_min_ade_candidate_index` uses ground-truth ADE. This field is only for evaluation and is not a deployment selector.

Use this command to make one metrics file after all shards are complete:

```bash
python src/export_candidate_metrics.py \
  --records-dir results/candidates_k6/records \
  --output results/candidates_k6/candidate_metrics.csv
```

### Environment Setup

Use these commands to create the environment:

```bash
conda env create -f environment.yml
conda activate alpamayo-r1-research
python -m pip install --no-build-isolation -r requirements/flash-attn.txt
python -m pip install --no-deps -r requirements/alpamayo-source.txt
python -m unittest discover -s tests -v
```

The separate installation of `flash-attn` is necessary. The build process requires an installed version of Torch.

### SLURM Operation

Use one visible GPU for each task. Use a zero-based SLURM array to process multiple clips.

Do not use the same shard index and output directory for concurrent tasks.

Use this command for a short test:

```bash
mkdir -p logs
export ALPAMAYO_PYTHON="$CONDA_PREFIX/bin/python"
export PARTITION=<gpu-partition>
NUM_CANDIDATES=1 MAX_CLIPS=1 sbatch --partition="$PARTITION" --array=0-0 \
  --output=logs/%x-%A_%a.out \
  slurm/run_generate_candidates.sh
```

Remove the `NUM_CANDIDATES` and `MAX_CLIPS` values for a full run. Select an array size that is correct for the cluster capacity.

Use `hf auth login` one time before an online run. As an alternative, use the scheduler secret system to supply `HF_TOKEN`.

Set `HF_HUB_OFFLINE=1` only when the cache contains all required files.

The SLURM script checks these conditions before model load:

- The project path is available.
- The clip file is available.
- The Python and package versions are correct.
- The Hugging Face credentials are available for an online run.
- The task has exactly one visible GPU.

The script accepts these optional environment variables:

- `ALPAMAYO_PROJECT_ROOT`
- `CLIP_PARQUET`
- `OUTPUT_DIR`
- `GPU_MEMORY`
- `CPU_MEMORY`

The process resumes only when an existing artifact has the correct configuration and identity. Set `OVERWRITE=1` to replace an incompatible artifact.

## Recorded-Future LiDAR Oracle

The LiDAR oracle is an evaluation tool. It compares each trajectory candidate with LiDAR data from later times.

The tool makes bird's-eye-view (BEV) grids. A BEV grid is a two-dimensional map around the ego vehicle.

The tool does these operations:

1. Correct each LiDAR point for ego motion.
2. Transform each point to the ego frame at the planning timestamp.
3. Make past and future occupancy grids.
4. Move the vehicle footprint along each trajectory candidate.
5. Calculate collision, clearance, observed-space, and out-of-bounds values.
6. Make an evaluation rank for the candidates.

The tool treats unobserved space and out-of-bounds space as unknown. A candidate with unknown exposure ranks after each fully observed, in-bounds candidate.

The oracle is not an interactive simulator. Other road users follow the recorded ego action and do not react to the trajectory candidates.

Read [`docs/lidar_world_oracle.md`](docs/lidar_world_oracle.md) for the coordinate equation, file schema, commands, and limits.

## Predictive BEV World Model

The world model uses past BEV grids to calculate future occupancy probabilities. A convolutional gated recurrent unit processes the time sequence.

The world-model process has these parts:

- Dataset groups that do not share a LiDAR source chunk
- Checkpoints that validate and restore the state at an epoch boundary
- A persistence baseline
- Metrics that use only observed cells
- Output files that identify all source files
- SLURM scripts that use one GPU

The candidate selector uses only predicted occupancy and trajectory data. It does not use the recorded-future occupancy or the oracle rank.

The repository does not contain a trained result. The acceptance limits in the implementation guide are test requirements, not results.

Read [`docs/bev_world_model.md`](docs/bev_world_model.md) for the data contract, commands, metrics, and acceptance limits.

## References

- [Alpamayo-R1 model card](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- [PhysicalAI Autonomous Vehicles dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- [Alpamayo source](https://github.com/NVlabs/alpamayo)
