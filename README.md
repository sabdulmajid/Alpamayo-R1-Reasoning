# Auditing Alpamayo-R1 Planning

NVIDIA's Alpamayo-R1-10B reads four recent camera streams and the vehicle's
recent motion. It then explains a driving decision and plans the vehicle's next
6.4 seconds. This repository tests the public model at the link between
explanation, motion, and future road occupancy.

[Paper](https://arxiv.org/abs/2511.00088) ·
[Model](https://huggingface.co/nvidia/Alpamayo-R1-10B) ·
[Official code](https://github.com/NVlabs/alpamayo) ·
[Public data](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)

## What Alpamayo-R1 Does

Alpamayo-R1 is a vision-language-action model for difficult driving cases. Its
Cosmos-Reason backbone writes a **Chain of Causation**. This text links visible
road facts to an intended response. A diffusion decoder then produces 64 future
vehicle positions at 10 Hz.

NVIDIA trained the model on 80,000 hours of internal driving data and 700,000
structured reasoning traces. The paper also uses reinforcement learning to
improve agreement between words and motion. The public v1.0 checkpoint contains
the supervised training stage, but not the paper's reinforcement-learning
weights.

The core idea is important: an explanation is useful only when the planned
motion follows it. This repository turns that link into a test on public data.

## What We Built

We did not train Alpamayo-R1. We evaluated its released checkpoint at a pinned
revision on 2,000 clips from the PhysicalAI-AV dataset. The project adds:

- A rule-based audit of stated intent against generated motion
- Six repeatable Alpamayo-R1 paths per clip, or 12,000 paths in total
- A 339,910-parameter LiDAR occupancy model trained from random weights
- A held-out test of forecast quality and path exposure

The occupancy model is the **world model** in this project. It receives three
LiDAR maps from the previous second. It predicts LiDAR occupancy at 0.5, 1, 2,
3, 4, and 6 seconds. Recorded future LiDAR supplies the training target. This is
a focused scene forecast, not a general simulator or a crash predictor.

## What We Found

**The audit found a measurable reasoning-action gap.** Under the documented
action-class rules, 833 of 2,000 clips had a mismatch score of at least 0.6.
Curved roads can affect the lateral score because the test uses the vehicle
coordinate frame. The result is an audit signal, not an unsafe-plan rate.

**The world model learned useful scene motion.** The split used 1,612 training,
180 validation, and 208 test clips. Source recording chunks do not cross the
splits. On held-out clips, the model beat a baseline that copies the current
map at every forecast time through six seconds. From 0.5 to 2 seconds, occupied
cell ranking improved by 0.217, probability error fell by 0.0259, and occupancy
overlap improved by 0.0386. Each paired 95 percent interval excludes zero.

**The six paths contain useful alternatives.** An evaluation-only oracle used
recorded future LiDAR to choose a path. Its geometric exposure was 0.01685,
compared with 0.03199 for the first path. This is a 47 percent reduction.
Exposure counts observed vehicle-footprint cells that contain future LiDAR
returns. It is not crash probability. The forecast-based selector scored
0.03868. This localizes the engineering gap: forecast quality alone does not
produce good path selection.

The result is a reproducible audit from stated reason to planned motion to
future scene state. Researchers can replace any one of these components while
keeping the same split, artifact checks, and evaluation.

## Evidence and Code

- [Methods, metric definitions, and complete results](docs/benchmark_results.md)
- [Candidate generation and SLURM runbook](docs/candidate_generation.md)
- [World-model design and training](docs/bev_world_model.md)
- [Recorded-future LiDAR oracle](docs/lidar_world_oracle.md)
- [Machine-readable results and artifact hashes](reports/world_model_benchmark_2k.json)

Two NVIDIA RTX A4500 GPUs trained independent seeds for 30 epochs. The runs
finished in 468 and 469 seconds. A third GPU job generated held-out forecasts.
All 122 tests pass.
