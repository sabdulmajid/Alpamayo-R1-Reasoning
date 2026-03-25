# Alpamayo-R1 Reasoning–Action Mismatch Study

This project asks one technical question: when Alpamayo-R1 explains its driving decision in language, does its predicted trajectory execute the same decision?

The model produces two outputs for the same scene:
- a chain-of-causation (CoC) text explanation
- a 6.4-second future trajectory (64 waypoints)

Mismatch is the gap between those two outputs.

## What Was Tested

### End-to-end test flow (simple example)

Given one clip at time \(t_0\):

1. The model sees multi-camera context and ego-motion history.
2. It generates text such as “slow down for red light” or “nudge left for clearance.”
3. It generates future waypoints that imply actual executed behavior (decelerating, straight, lane shift, lane change).
4. We parse text intent and trajectory execution independently and compare them.

If text says “slow down” but trajectory accelerates, mismatch is high. If both agree, mismatch is low.

### Experimental design

- Model: Alpamayo-R1-10B in native bf16 (no quantization)
- Compute constraint: single NVIDIA RTX A4500 (20GB VRAM)
- Dataset: PhysicalAI-Autonomous-Vehicles test split
- Evaluation set: 2,000 clips, stratified by hour-of-day (balanced across 5 buckets)
- Runtime strategy: batched + resumable execution to complete long runs under strict single-GPU limits

## How Mismatch Was Measured

Mismatch was computed on two axes so failure modes are visible instead of hidden in one score.

1. **Longitudinal axis (speed intent)**
	- Text intent classes: stop, slow_down, maintain, accelerate
	- Trajectory execution classes: stopped, decelerating, constant_speed, accelerating

2. **Lateral axis (steering intent)**
	- Text intent classes: hold_lane, nudge_left/right, lane_change_left/right
	- Trajectory execution classes: straight, shift_left/right, lane_change_left/right

Each axis uses an explicit compatibility matrix. Final mismatch score is \(1 - \text{compatibility}\), where 0 means full agreement and 1 means contradiction.

## What Was Done to Make Results Valid

Early runs exposed two important methodological issues, both fixed before the final 2,000-clip run:

1. **Timestamp validity issue**
	- Initial clip sampling used a wider \(t_0\) range and produced many invalid samples because camera coverage was tighter than egomotion coverage.
	- Fix: enforce a camera-safe \(t_0\) window and resample.

2. **Text parser undercoverage issue**
	- Early parser missed common CoC phrasing (“keep distance,” “adapt speed,” “yield,” curve-following language).
	- Fix: expand intent patterns, then re-score to confirm the previous “unclassified” bucket was mostly parser blind spots.

## Final Results (2,000 clips)

All metrics below are computed on the final 2,000-clip run.

- Clips processed: **2,000 / 2,000**
- Valid scored results: **2,000 / 2,000**
- Runtime failures: **0**
- Mean mismatch score: **0.4991** (95% CI: **0.4839–0.5143**, SD: 0.3463)
- Mean ADE: **1.9391 m** (95% CI: **1.8556–2.0226**, SD: 1.9047)
- Mean longitudinal match: **0.6125** (95% CI: **0.5963–0.6288**)
- Mean lateral match: **0.3776** (95% CI: **0.3639–0.3913**)

Mismatch severity distribution:

- Consistent (`score < 0.3`): **531 / 2000 = 26.55%** (95% CI: 24.61–28.49%)
- Partial (`0.3 <= score < 0.6`): **636 / 2000 = 31.80%** (95% CI: 29.76–33.84%)
- Severe (`score >= 0.6`): **833 / 2000 = 41.65%** (95% CI: 39.49–43.81%)

## Technical Deep-Dive: How Conclusions Were Derived

### 1) Axis asymmetry is large and statistically stable

Observed difference between longitudinal and lateral match:

- `mean(long_match - lat_match) = 0.2349`
- 95% CI: **0.2131–0.2568**

This gap excludes zero by a wide margin, so the two axes are not behaving similarly. The lateral axis is systematically harder to align than the speed axis in this setup.

### 2) “Mismatch” is not only a tail event

Severe mismatch is **41.65%**, not a rare outlier bucket. Combined partial+severe is **73.45%**. This directly establishes that disagreement is common under this scoring protocol.

### 3) Scenario differences are real but second-order

Balanced design gives 400 clips per hour bucket. Severe rates:

- Midday: **38.50%**
- Morning: **41.50%**
- Afternoon: **42.25%**
- Night: **42.75%**
- Evening: **43.25%**

Interpretation: mismatch exists in all buckets; time-of-day shifts are modest relative to the global mismatch level.

### 4) Error anatomy from intent-execution cross-tabs

Most frequent longitudinal pairings:

- `maintain -> constant_speed`: 540
- `maintain -> accelerating`: 316
- `slow_down -> decelerating`: 204
- `slow_down -> constant_speed`: 171

Most frequent lateral pairings:

- `hold_lane -> lane_change_right`: 198
- `hold_lane -> lane_change_left`: 175
- `hold_lane -> shift_right`: 160
- `hold_lane -> shift_left`: 157
- `hold_lane -> straight`: 134

These cross-tabs are the basis for the geometry-confound claim: the dominant lateral contradiction pattern is not random; it is concentrated in `hold_lane` text against non-straight ego-frame execution classes.

## Conclusions (Supported by Current Data)

1. **Under this protocol, reasoning-action mismatch is quantitatively high** (mean 0.499; severe 41.65%).

2. **Mismatch is axis-dependent**: longitudinal alignment is materially higher than lateral alignment by ~0.235 absolute points (tight CI).

3. **The current lateral estimate is likely upward-biased by representation effects**: dominant contradiction pairs are consistent with ego-frame curvature aliasing (lane-following on curved roads mapped to lateral shift/change labels).

4. **Therefore, the study has already established a robust discrepancy signal, but not a pure causal decomposition** between true decision inconsistency and coordinate-frame artifact.

## Data-Backed Next Steps (Decision-Oriented)

### Next Step A — Curvature-normalized lateral scoring (highest priority)

What to add:
- derive road-following baseline from trajectory curvature and heading change
- reclassify lateral execution relative to that baseline (instead of raw ego-y displacement)

Decision criterion:
- If severe mismatch drops by **>= 10 absolute points** and most reduction comes from `hold_lane -> lane_change_* / shift_*`, current lateral contradiction is mostly representational.
- If reduction is **< 5 points**, contradiction is mostly model-behavioral.

### Next Step B — Parser ablation with frozen trajectories

What to run:
- score same 2,000 trajectories with parser variants (strict / current / expanded)

Decision criterion:
- If severe mismatch variance across parser versions is **> 5 points**, conclusions are parser-sensitive and must be reported with parser uncertainty.
- If **<= 2 points**, conclusions are parser-robust.

### Next Step C — Outcome coupling test (mismatch vs trajectory quality)

Current signal:
- corr(mismatch, ADE) = **0.1234** (weak positive)
- mean ADE by band: consistent 1.810m, partial 1.693m, severe 2.209m

What to do:
- run controlled analysis (stratify by scenario + speed regime)

Decision criterion:
- If severe retains significantly higher ADE after stratification, mismatch is safety-relevant.
- If effect collapses, mismatch is mostly semantic labeling noise.

### Next Step D — Human audit on targeted slice

What to sample:
- 200 clips: top 100 severe `hold_lane` contradictions + 100 matched controls

Decision criterion:
- Annotator agreement identifies whether each case is true contradiction vs curvature artifact.
- This becomes the calibration set for metric correction and paper claims.

## Threats to Validity

### 1) Construct validity (does the metric measure what we claim?)

Primary risk: lateral mismatch partly captures coordinate/frame effects, not only behavioral contradiction.

Evidence from current run:
- dominant conflict mass is `hold_lane -> lane_change_* / shift_*`
- lateral execution classes are heavily non-straight (`lane_change_right=542`, `lane_change_left=456`, `shift_right=316`, `shift_left=288`, `straight=398`)

Consequence:
- current lateral mismatch is an upper-bound estimate of contradiction, not a geometry-corrected estimate.

### 2) Internal validity (could pipeline artifacts drive findings?)

Known mitigations already applied:
- camera-safe `t0` sampling to avoid invalid timestamp windows
- parser expansion to reduce false “unclassified” labels
- deterministic batch processing with complete run (2,000/2,000, zero runtime failures)

Residual risk:
- parser still defines intent boundaries; some semantic ambiguity remains unavoidable without human labels.

### 3) External validity (how far do these results generalize?)

Current scope limitations:
- one model family (Alpamayo-R1-10B)
- one dataset family (PhysicalAI-AV test split)
- one hardware/runtime regime (single A4500, bf16 + offload strategy)

Consequence:
- conclusions are valid for this protocol and setup; cross-model and cross-dataset claims require replication.

### 4) Statistical conclusion validity

Strengths:
- balanced design (400 clips per hour bucket)
- narrow CIs on core metrics at N=2000

Limits:
- effect interpretation is sensitive to class definition and compatibility matrix choices.
- weak mismatch–ADE correlation (`r=0.1234`) means mismatch is not yet a strong standalone proxy for trajectory quality.

### Reviewer-facing summary

What is strong:
- high mismatch prevalence under an explicit, reproducible scoring protocol
- stable axis asymmetry (longitudinal > lateral) with tight uncertainty

What is not yet closed:
- causal separation of true reasoning inconsistency vs representation-induced lateral inflation
- portability of effect sizes beyond this model/dataset protocol

## References

- Model card: https://huggingface.co/nvidia/Alpamayo-R1-10B
- Dataset: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
- Alpamayo code: https://github.com/NVlabs/alpamayo