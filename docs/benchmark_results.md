# Benchmark Results

This document contains the detailed results for the two project studies. The
main README gives a short project overview.

## Evaluation Scope

| Item | Value |
| --- | --- |
| Model | Alpamayo-R1-10B |
| Dataset | PhysicalAI Autonomous Vehicles |
| Evaluation pool | 2,000 clips |
| Candidate paths | 6 for each clip; 12,000 total |
| Forecast horizons | 0.5, 1, 2, 3, 4, and 6 seconds |
| Forecast model | 339,910 parameters; trained from random weights |
| GPU | NVIDIA RTX A4500 |

The experiment pins the model, dataset, and source code revisions. The
machine-readable report contains the exact revisions and artifact hashes.

## Reasoning and Action Agreement

Alpamayo-R1 produces a Chain of Causation explanation and a 6.4-second
trajectory. The test assigns longitudinal and lateral intent classes to the
text. It assigns action classes to the trajectory. A compatibility matrix then
calculates a mismatch score.

A mismatch score of 0 means full agreement. A score of 1 means a contradiction
under the documented class rules.

| Metric | Result |
| --- | ---: |
| Valid clips | 2,000 of 2,000 |
| Mean mismatch | 0.4991; 95 percent interval 0.4839 to 0.5143 |
| Mismatch standard deviation | 0.3463 |
| Mean average displacement error | 1.9391 m; interval 1.8556 m to 2.0226 m |
| Mean longitudinal match | 0.6125; interval 0.5963 to 0.6288 |
| Mean lateral match | 0.3776; interval 0.3639 to 0.3913 |

| Mismatch level | Score | Clips | Rate |
| --- | --- | ---: | ---: |
| Consistent | Less than 0.3 | 531 | 26.55 percent |
| Partial | 0.3 to less than 0.6 | 636 | 31.80 percent |
| Severe | 0.6 or more | 833 | 41.65 percent |

The longitudinal match is 0.2349 higher than the lateral match. Its paired
interval is 0.2131 to 0.2568.

The lateral measurement uses motion in the ego coordinate frame. A curved road
can look like a lateral maneuver in this frame. For this reason, the lateral
mismatch value can include coordinate-representation error.

## Occupancy Forecast Training

The occupancy model receives three LiDAR maps from -1.0, -0.5, and 0 seconds.
It predicts occupied cells at six future times. Recorded future LiDAR supplies
the training target.

The 2,000-clip pool has three internal sets:

| Set | Clips | LiDAR source chunks |
| --- | ---: | ---: |
| Training | 1,612 | 486 |
| Validation | 180 | 55 |
| Test | 208 | 67 |

No source chunk occurs in more than one set. Validation loss selected seed 2027
at epoch 16. Test metrics did not select the checkpoint.

The benchmark compares the learned model with a copy-current baseline. The
baseline repeats the last observed occupancy map at every future time.

## Forecast Measures

- **Average precision** checks if occupied cells are near the top of the model's
  probability ranking. Higher is better.
- **Intersection over union** measures the overlap between predicted and
  recorded occupied areas. Higher is better.
- **Brier score** is the mean squared probability error. Lower is better.

These measures are not percentages of correct cells. The test calculates each
measure for each clip and horizon. It then gives each clip equal weight.

The primary result uses the 0.5-second, 1-second, and 2-second horizons:

| Measure | Learned | Copy-current | Difference | Paired 95 percent interval |
| --- | ---: | ---: | ---: | ---: |
| Average precision | 0.81543 | 0.59847 | +0.21696 | +0.20922 to +0.22485 |
| Brier score | 0.09813 | 0.12401 | -0.02589 | -0.03008 to -0.02189 |
| IoU at a 0.5 threshold | 0.59933 | 0.56072 | +0.03861 | +0.03033 to +0.04664 |

The learned forecast improves all three measures. It also improves each measure
at every individual horizon from 0.5 through 6 seconds.

![Held-out occupancy forecast two seconds ahead](assets/world_model_median_example.png)

The figure uses one held-out clip at the 2-second horizon. A script selected the
clip closest to the median short-horizon Brier improvement before plotting.
White cells contain recorded LiDAR endpoints. Color in the other panels shows
forecast occupancy probability. Black cells were not observed and were not
scored. The blue rectangle marks the ego vehicle position at planning time.

## Candidate-Path Audit

The path audit places the physical vehicle footprint along each candidate. It
then measures overlap with recorded LiDAR occupancy at the same future times.

For example, assume that 100 observed footprint-cell checks occur across the
future times. If 4 checks contain a LiDAR endpoint, collision exposure is 0.04.
This value is a geometric overlap rate. It is not a crash probability.

| Policy | Collision exposure | Clips with overlap | ADE | Outside-map fraction |
| --- | ---: | ---: | ---: | ---: |
| Candidate 0 | 0.03199 | 30.29 percent | 1.8436 m | 0.10096 |
| Learned selector | 0.03868 | 34.13 percent | 2.3110 m | 0.10176 |
| Copy-current selector | 0.04436 | 36.06 percent | 2.3549 m | 0.09535 |
| Recorded-future oracle | 0.01685 | 16.83 percent | 1.7311 m | 0.09135 |

Candidate 0 is the first stored stochastic rollout and the fixed reference for
this test. The learned selector improved on the copy-current selector. It did
not improve on candidate 0. Its exposure difference from candidate 0 has a
paired interval of -0.00355 to +0.01779.

The recorded-future oracle uses future LiDAR to select from the six candidates.
It reduced mean exposure by 47 percent compared with candidate 0. It is an
evaluation reference, not a deployable selector. This result shows measurable
headroom in the candidate set.

## Map Coverage

The bird's-eye-view grid covers 20 m behind the planning position, 80 m ahead,
and 40 m on each side. At 6 seconds, candidate 0 was partly outside the grid in
745 of the 2,000 clips. In 590 clips, all six vehicle footprints were fully
outside the grid.

An outside-grid location is unknown. It is not an empty or safe location.

## Uncertainty and Reproduction

The paired intervals use 10,000 bootstrap samples of the 67 held-out source
chunks. They measure variation across held-out chunks. They do not measure the
full variation from repeated training. The run used two training seeds.

The two training jobs ran together on two RTX A4500 GPUs. They completed 30
epochs in 468 and 469 seconds. A third GPU job made the 208 held-out forecasts.
CPU arrays built the LiDAR labels and ran the statistical evaluation.

All downstream jobs ended with exit code 0. Their error logs were empty. The
repository test suite contains 122 passing tests.

Read [`../reports/world_model_benchmark_2k.json`](../reports/world_model_benchmark_2k.json)
for full-precision results, immutable revisions, and artifact hashes.
