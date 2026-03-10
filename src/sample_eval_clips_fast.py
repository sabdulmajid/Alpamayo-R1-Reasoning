"""
Sample a large stratified eval set from PhysicalAI-AV test split.

For large N (2000+), skip per-clip egomotion validation — the safe t0
window [2s, 13s] is well within the guaranteed camera/egomotion coverage
for all valid clips. Invalid clips fail gracefully at inference time.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import physical_ai_av

SAFE_T0_MIN_US = 2_000_000
SAFE_T0_MAX_US = 13_000_000

HOUR_BINS = [0, 7, 11, 15, 19, 24]
HOUR_LABELS = ["night", "morning", "midday", "afternoon", "evening"]


def sample_clips(n_clips: int, seed: int = 99) -> pd.DataFrame:
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    clip_index = avdi.clip_index
    avdi.download_metadata()

    # Join metadata for hour_of_day / country
    import os, time
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    # Wait for metadata download to land (up to 30s)
    meta_path = None
    for _ in range(30):
        candidates = list(hf_home.rglob("data_collection.parquet"))
        if candidates:
            meta_path = candidates[0]
            break
        if avdi.cache_dir:
            candidates = list(Path(avdi.cache_dir).rglob("data_collection.parquet"))
            if candidates:
                meta_path = candidates[0]
                break
        time.sleep(1)

    if meta_path:
        dc = pd.read_parquet(meta_path)
        clip_index = clip_index.join(dc, how="left")
        print(f"Joined metadata from {meta_path}")
    else:
        raise RuntimeError("Could not find data_collection.parquet after download")

    test_clips = clip_index[
        (clip_index["split"] == "test") & (clip_index["clip_is_valid"])
    ].copy()
    print(f"Test split: {len(test_clips)} valid clips")

    # Stratify by hour buckets
    test_clips["hour_bucket"] = pd.cut(
        test_clips["hour_of_day"], bins=HOUR_BINS, labels=HOUR_LABELS, right=False
    )

    rng = np.random.default_rng(seed)
    per_bucket = n_clips // len(HOUR_LABELS)

    frames = []
    for label in HOUR_LABELS:
        bucket_clips = test_clips[test_clips["hour_bucket"] == label]
        n_pick = min(per_bucket, len(bucket_clips))
        picked = bucket_clips.sample(n=n_pick, random_state=int(rng.integers(1e9)))
        t0_values = rng.integers(SAFE_T0_MIN_US, SAFE_T0_MAX_US, size=n_pick)
        df_bucket = pd.DataFrame({
            "clip_id": picked.index,
            "t0_us": t0_values,
            "hour_bucket": label,
            "country": picked["country"].values if "country" in picked.columns else "unknown",
        })
        frames.append(df_bucket)
        print(f"  {label}: {n_pick} clips")

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\nTotal: {len(df)} clips")
    print(f"t0 range: {df['t0_us'].min()/1e6:.2f}s - {df['t0_us'].max()/1e6:.2f}s")
    print(df["hour_bucket"].value_counts().sort_index().to_string())
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clips", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output", type=str, default="data/eval_clips_2k.parquet")
    args = parser.parse_args()

    df = sample_clips(args.n_clips, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
