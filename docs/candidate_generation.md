# Candidate Generation

This process uses Alpamayo-R1-10B to make six trajectory candidates for each
driving clip. Each candidate comes from a separate model rollout.

## Configuration

| Item | Value |
| --- | ---: |
| Candidate count | 6 |
| Samples in each rollout | 1 |
| Top-p | 0.98 |
| Temperature | 0.6 |
| Maximum generation length | 256 tokens |

The process pins the model, dataset, and Alpamayo source to immutable revisions.
It calculates one stable seed for each clip and one seed for each candidate.

The validated cluster configuration uses one NVIDIA RTX A4500 for each task.
CPU offload keeps the model within the available GPU memory.

## Output

Each completed clip has these files:

| Path | Content |
| --- | --- |
| `artifacts/<key>.npz` | Trajectories, sample times, seeds, text, ground truth, and clip identity |
| `records/<key>.json` | Configuration, metrics, seeds, and the artifact path |
| `manifests/shard-xxxxx-of-yyyyy.jsonl` | Completed records for one shard |

The process writes files atomically. Each SLURM task writes a separate manifest.
Resume checks verify the configuration, clip identity, and artifact content.

The field `oracle_min_ade_candidate_index` uses the recorded path. Use this field
for evaluation only. Do not use it to select a path during deployment.

Export one candidate-metrics file after all shards finish:

```bash
python src/export_candidate_metrics.py \
  --records-dir results/candidates_k6/records \
  --output results/candidates_k6/candidate_metrics.csv
```

## Full Environment

Create the full project environment:

```bash
conda env create -f environment.yml
conda activate alpamayo-r1-research
python -m pip install --no-build-isolation -r requirements/flash-attn.txt
python -m pip install --no-deps -r requirements/alpamayo-source.txt
python -m unittest discover -s tests -v
```

Install `flash-attn` after Torch. Its build process requires Torch.

Run `hf auth login` before an online model run. You can also use the scheduler
secret system to supply `HF_TOKEN`.

## SLURM Example

Use one visible GPU for each task. Use a zero-based array to process clips.
Do not use the same shard index and output directory in concurrent tasks.

Run one clip with one candidate:

```bash
mkdir -p logs
export ALPAMAYO_PYTHON="$CONDA_PREFIX/bin/python"
export PARTITION=<gpu-partition>
NUM_CANDIDATES=1 MAX_CLIPS=1 sbatch --partition="$PARTITION" --array=0-0 \
  --output=logs/%x-%A_%a.out \
  slurm/run_generate_candidates.sh
```

Remove `NUM_CANDIDATES` and `MAX_CLIPS` for a complete run. Select an array size
that matches the cluster capacity.

The SLURM script checks these items before model load:

- Project and clip paths
- Python and package versions
- Hugging Face credentials for an online run
- Exactly one visible GPU

The script accepts these optional variables:

- `ALPAMAYO_PROJECT_ROOT`
- `CLIP_PARQUET`
- `OUTPUT_DIR`
- `GPU_MEMORY`
- `CPU_MEMORY`

Set `OVERWRITE=1` only when you intend to replace an incompatible artifact.
