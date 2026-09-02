# Recorded-Future LiDAR World Oracle

The oracle evaluates Alpamayo trajectory candidates against LiDAR observed after
the planning timestamp. It is an evaluation and label-generation tool for
world-aware reranking. It is not used as planner input at inference time.

## Coordinate frames

Every output grid and trajectory uses the ego rig frame at `t0`: x points
forward, y points left, and z points up. PhysicalAI-AV supplies sensor-to-rig
extrinsics and timestamped rig-to-world egomotion. Each LiDAR return uses its
absolute per-point timestamp:

```text
p_ego@t0 = inv(T_world_rig(t0))
             @ T_world_rig(point_timestamp)
             @ T_rig_lidar
             @ p_lidar
```

This compensates both the motion between spins and motion during a spin. The
implementation follows the PhysicalAI-AV transform direction and Draco fields
used by NVIDIA's
[NCore PAI converter](https://github.com/NVIDIA/ncore/blob/main/tools/data_converter/pai/converter.py).

## Occupancy construction

The builder samples history at `[-1.0, -0.5, 0.0]` seconds by default and future
targets at `[0.5, 1.0, 2.0, 3.0, 4.0, 6.0]` seconds. Both can be changed with
`--history-offsets` and `--horizons`. The nearest LiDAR spin midpoint must fall
within `--spin-tolerance` (0.2 s by default). The decoder reads Draco xyz and
the absolute per-point `timestamp` attribute. For every history and future spin
it then:

1. transforms returns into `ego_at_t0`;
2. removes returns inside the padded vehicle bounding box using the dataset's
   length, width, height, and rear-axle-to-center offset;
3. applies configurable range, ground-height, and maximum-height filters;
4. rasterizes endpoint hit counts and binary occupancy; and
5. approximates observed line of sight by rasterizing every Nth ray, controlled
   by `--observed-ray-stride`.

The default BEV covers x `[-20, 80)` m and y `[-40, 40)` m at 0.5 m per cell.
Array layout is `[horizon, row_y, column_x]`. The ground filter is intentionally
geometric, not semantic; use `--disable-ground-filter` for ablations.

## Candidate risk

Candidate positions are rear-axle poses in `ego_at_t0`. Heading is loaded from
`pred_yaw` or `pred_rot` when present. Older artifacts containing only
`pred_xyz` use the path tangent, with the nearest moving heading carried through
stationary samples. The output records this choice in `candidate_yaw_source`.

At every future occupancy horizon, the evaluator rasterizes the oriented dataset
vehicle footprint at the nearest candidate timestamp. Every requested horizon
must be within the candidate time range. It reports:

- `collision_exposure`: occupied cells divided by observed footprint cells;
- `collision_cells`: total occupied footprint cells across horizons;
- `conflict_horizons`: horizons containing at least one occupied footprint cell;
- `observed_fraction`: observed in-grid footprint cells divided by all in-grid
  footprint cells;
- `unobserved_horizons`: horizons with one or more unobserved in-grid footprint
  cells;
- `min_clearance_m`: minimum grid distance from a footprint to an occupied cell;
- `first_conflict_s`: first horizon with footprint overlap, or NaN;
- `out_of_bounds_horizons` and `out_of_bounds_fraction`: requested horizons
  where any part of the footprint leaves the BEV, and their fraction; and
- `candidate_oracle_rank`: a deterministic safety ordering, where zero is best.

Out-of-grid and unobserved space are unknown, not free. A candidate with
out-of-bounds exposure ranks after candidates whose complete footprints stay in
the BEV. The next ranking terms prefer fewer unobserved horizons and higher
observed coverage. Remaining ties prefer fewer conflicts, fewer collision
cells, lower exposure, and larger clearance. `oracle_safest_idx` is an
evaluation label, not a deployable selection policy.

## Running one shard

Install the additional pinned dependencies:

```bash
python -m pip install -r requirements-world-oracle.txt
```

Then run a bounded smoke test. `--candidate-dir` is the output root produced by
`generate_candidates.py`; it must contain `records/*.json` and `artifacts/*.npz`.
The oracle resolves each hash-named artifact through its per-clip record, then
checks the record's clip identity and configuration fingerprint against values
embedded in the NPZ. It also binds each oracle artifact to the exact candidate
file bytes with SHA-256, so regenerating samples invalidates stale resume data.

```bash
python src/lidar_world_oracle.py \
  --clip-parquet data/eval_clips_2k.parquet \
  --candidate-dir results/candidates_k6 \
  --output-dir results/lidar_world_oracle \
  --limit 10
```

The dataset is pinned to revision
`2ae73f49ffd2b5db43b404201beb7b92889f7afc`. Override it only when deliberately
regenerating all derived artifacts.

For SLURM, export the Python executable and any non-default paths before
submission. The array assigns an entire PhysicalAI-AV source chunk to one shard
using `chunk_id % num_shards`, avoiding cross-shard archive overlap.

```bash
PYTHON_BIN=/path/to/env/bin/python \
CANDIDATE_DIR=/path/to/candidate_output_root \
sbatch --export=ALL slurm/build_lidar_world_oracle.sh
```

The wrapper uses the currently accessible `dualcard` partition but does not
request a GPU; decoding and rasterization are CPU workloads. Override the
partition at submission time if cluster availability changes.

## Output contract

Each clip is written atomically under `<output>/clips/` with a SHA-256-derived
filename; raw clip IDs never become path components. Files contain no pickled
objects and can be loaded with `allow_pickle=False`.

| Key | Type and shape | Meaning |
| --- | --- | --- |
| `past_occupancy` | `uint8 [P,H,W]` | History endpoint occupancy at non-positive offsets |
| `past_observed` | `uint8 [P,H,W]` | History ray-observed cells |
| `past_occupancy_count` | `uint16 [P,H,W]` | History endpoint hit counts |
| `history_offsets_s` | `float32 [P]` | Requested history times relative to `t0` |
| `history_spin_timestamps_us` | `int64 [P]` | Selected history spin midpoints |
| `occupancy` | `uint8 [N,H,W]` | Future endpoint occupancy |
| `observed` | `uint8 [N,H,W]` | Approximate ray-observed cells |
| `occupancy_count` | `uint16 [N,H,W]` | Endpoint hit counts, saturated at 65,535 |
| `horizons_s` | `float32 [N]` | Requested times relative to `t0` |
| `spin_timestamps_us` | `int64 [N]` | Selected spin midpoint timestamps |
| `candidate_xyz` | `float32 [K,T,3]` | Candidate trajectories in `ego_at_t0` |
| `candidate_yaw` | `float32 [K,T]` | Candidate heading |
| `candidate_times_s` | `float32 [T]` | Candidate times relative to `t0` |
| `candidate_*` risk arrays | `[K]` | Collision, coverage, unknown-space, clearance, conflict-time, and rank values |
| `vehicle_dimensions_m` | `float32 [4]` | Length, width, height, rear-axle-to-center offset |
| `bev_*` | scalar | Metric bounds and resolution |
| `oracle_config_*` | scalar strings | Canonical configuration and SHA-256 fingerprint |
| `candidate_config_fingerprint` | scalar string | Candidate-generation configuration identity |
| `candidate_artifact_sha256` | scalar string | Exact candidate NPZ content identity |

One atomic JSONL manifest is written per shard. Its `artifact_path` is relative
to the manifest and can be consumed directly by the world-model data loader. It
also includes `chunk_id`, clip identity, candidate count, pinned revision,
history/future times, configuration fingerprint, candidate content digest, yaw
source, and oracle index.
Downstream splits must group by `chunk_id` to prevent source-archive leakage.

`--resume` accepts an existing clip only when its schema, identity, dataset and
candidate configuration, exact candidate content digest, complete oracle
configuration, temporal axes, array values and shapes, and BEV geometry match
the current request. Resume also compares the saved candidate geometry with the
source artifact and recalculates all risk values. A mismatch fails rather than
mixing incompatible artifacts.

## Limitations

- Recorded-world replay is not interactive counterfactual simulation. Other
  actors followed the logged ego action, not each candidate action.
- Endpoint occupancy has no object identity, class, velocity, or free-space
  semantics. It must not be described as detection or tracking output.
- The height threshold is a simple ground proxy and can remove low obstacles or
  retain sloped-road returns.
- Visibility is a subsampled ray mask. It is useful for masking occupancy loss,
  but it is not a complete occlusion model.
- Collision and clearance are grid-quantized proxies. They are evaluation
  signals, not certified safety checks.
