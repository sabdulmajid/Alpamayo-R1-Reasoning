"""
Sample a stratified eval set from the PhysicalAI-AV dataset.

Samples clips from test split, stratified by hour-of-day (proxy for
lighting/scenario diversity). Uses CAMERA timestamp range (not egomotion)
to compute safe t0_us — cameras cover ~0-20s while egomotion extends
to ~134s, and the model needs images from all 4 cameras at t0.

Usage:
    python src/sample_eval_clips.py --n-clips 100 --output data/eval_clips.parquet
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import physical_ai_av


# The model needs at time t0:
#   history: 16 steps * 0.1s = 1.6s BEFORE t0  (egomotion)
#   images:  4 frames * 0.1s = 0.3s BEFORE t0  (cameras)
#   future:  64 steps * 0.1s = 6.4s AFTER t0   (egomotion)
#
# Camera data reliably covers ~0s to ~20s across the dataset.
# Egomotion extends to ~134s but the cameras are the binding constraint.
# Safe t0 window: [2.0s, 13.0s] — 1.6s history margin + 6.4s future + slack.
SAFE_T0_MIN_US = 2_000_000
SAFE_T0_MAX_US = 13_000_000


def sample_clips(n_clips: int, seed: int = 42, oversample_factor: int = 3) -> pd.DataFrame:
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    clip_index = avdi.clip_index
    avdi.download_metadata()

    meta_path = None
    if avdi.cache_dir:
        meta_path = next(Path(avdi.cache_dir).rglob("data_collection.parquet"), None)
    if meta_path is None:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        meta_path = next(hf_home.rglob("data_collection.parquet"), None)
    if meta_path:
        dc = pd.read_parquet(meta_path)
        clip_index = clip_index.join(dc, how="left")

    test_clips = clip_index[
        (clip_index["split"] == "test") & (clip_index["clip_is_valid"])
    ].copy()
    print(f"Test split: {len(test_clips)} valid clips")

    has_hour = "hour_of_day" in test_clips.columns

    # Oversample to account for clips we might skip during validation
    target = n_clips * oversample_factor

    if has_hour:
        bins = [0, 7, 11, 15, 19, 24]
        labels = ["night", "morning", "midday", "afternoon", "evening"]
        test_clips["hour_bucket"] = pd.cut(
            test_clips["hour_of_day"], bins=bins, labels=labels, right=False
        )
        per_bucket = max(1, target // len(labels))
        sampled = (
            test_clips.groupby("hour_bucket", observed=True)
            .sample(n=min(per_bucket, len(test_clips)), random_state=seed)
        )
        if len(sampled) > target:
            sampled = sampled.sample(n=target, random_state=seed)
    else:
        sampled = test_clips.sample(n=min(target, len(test_clips)), random_state=seed)

    print(f"Candidate pool: {len(sampled)} clips (oversampled {oversample_factor}x)")

    rng = np.random.default_rng(seed)
    rows = []
    bucket_counts = {l: 0 for l in labels} if has_hour else {}
    per_bucket_target = n_clips // len(labels) if has_hour else n_clips

    for clip_id in sampled.index:
        bucket = str(sampled.loc[clip_id, "hour_bucket"]) if has_hour else "all"
        if has_hour and bucket_counts.get(bucket, 0) >= per_bucket_target:
            continue

        try:
            # Light check: just verify egomotion covers the safe range
            # (avoids downloading camera data for 300 clips)
            eg = avdi.get_clip_feature(
                clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True
            )
            ts = eg.timestamps
            ego_min_us = int(ts[0])
            ego_max_us = int(ts[-1])

            # Egomotion must cover: (t0 - 1.6s) to (t0 + 6.4s)
            # With t0 up to SAFE_T0_MAX_US, ego must reach at least 13s + 6.4s = 19.4s
            # With t0 as low as SAFE_T0_MIN_US, ego must start before 2s - 1.6s = 0.4s
            if ego_max_us < SAFE_T0_MAX_US + 6_500_000:
                continue
            if ego_min_us > SAFE_T0_MIN_US - 1_700_000:
                continue

            # Random t0 within the safe camera window
            t0_us = int(rng.integers(SAFE_T0_MIN_US, SAFE_T0_MAX_US))

            row = {"clip_id": clip_id, "t0_us": t0_us}
            if has_hour:
                row["hour_bucket"] = bucket
            if "country" in sampled.columns:
                row["country"] = sampled.loc[clip_id, "country"]
            rows.append(row)

            if has_hour:
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        except Exception as e:
            print(f"  SKIP {clip_id}: {e}")

        if len(rows) >= n_clips:
            break

    df = pd.DataFrame(rows)
    print(f"\nFinal eval set: {len(df)} clips")
    if has_hour and "hour_bucket" in df.columns:
        print(df["hour_bucket"].value_counts().to_string())
    print(f"t0_us range: {df['t0_us'].min()/1e6:.2f}s - {df['t0_us'].max()/1e6:.2f}s")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clips", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/eval_clips.parquet")
    args = parser.parse_args()

    df = sample_clips(args.n_clips, seed=args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
