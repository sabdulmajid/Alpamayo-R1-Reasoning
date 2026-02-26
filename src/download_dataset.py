"""
Dataset download and preprocessing for PhysicalAI-AV.
Downloads a strategic subset focusing on stress scenarios.
"""

import argparse
import os
import json
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

# Ensure cache is on NFS
os.environ["HF_HOME"] = "/mnt/slurm_nfs/a6abdulm/.cache/huggingface"

DATASET_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"


def list_available_files():
    """List files in the dataset to understand structure."""
    print("Listing available files in dataset...")
    try:
        files = list_repo_files(DATASET_ID, repo_type="dataset")
        print(f"Found {len(files)} files total")
        
        # Categorize files
        categories = {}
        for f in files:
            parts = f.split("/")
            if parts:
                cat = parts[0]
                categories[cat] = categories.get(cat, 0) + 1
        
        print("\nFile categories:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} files")
        
        return files
    except HfHubHTTPError as e:
        print(f"Error accessing dataset: {e}")
        print("Make sure you've accepted the license agreement at:")
        print(f"  https://huggingface.co/datasets/{DATASET_ID}")
        return []


def download_metadata(output_dir: Path):
    """Download metadata files for clip filtering."""
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_files = [
        "metadata/data_collection.parquet",
        "metadata/sensor_presence.parquet",
    ]
    
    print("\nDownloading metadata files...")
    for f in metadata_files:
        try:
            local_path = hf_hub_download(
                repo_id=DATASET_ID,
                filename=f,
                repo_type="dataset",
                local_dir=output_dir,
            )
            print(f"  Downloaded: {f}")
        except Exception as e:
            print(f"  Failed to download {f}: {e}")
    
    return metadata_dir


def filter_clips_by_scenario(
    metadata_dir: Path,
    scenarios: list[str],
    num_clips: int
) -> list[str]:
    """
    Filter clips based on desired scenarios.
    Returns list of clip_ids to download.
    """
    import pandas as pd
    
    data_collection_path = metadata_dir.parent / "metadata" / "data_collection.parquet"
    
    if not data_collection_path.exists():
        print(f"Metadata not found at {data_collection_path}")
        return []
    
    print(f"\nLoading metadata from {data_collection_path}...")
    df = pd.read_parquet(data_collection_path)
    print(f"Total clips in dataset: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # The metadata contains: clip_id, country, month, hour_of_day, platform_class
    # We need to sample diverse scenarios based on available fields
    
    selected_clips = []
    clips_per_scenario = num_clips // len(scenarios) if scenarios else num_clips
    
    # Map scenario names to filtering logic
    scenario_filters = {
        "night": lambda df: df[df["hour_of_day"].isin([0, 1, 2, 3, 4, 5, 20, 21, 22, 23])],
        "clear": lambda df: df[df["hour_of_day"].isin([10, 11, 12, 13, 14, 15])],  # Daytime proxy
        "rain": lambda df: df,  # Need weather metadata - sample randomly for now
        "fog": lambda df: df,   # Need weather metadata - sample randomly for now
        "heavy_traffic": lambda df: df,  # Need traffic metadata - sample randomly
        "cut_in": lambda df: df,  # Need label metadata - sample randomly
    }
    
    for scenario in scenarios:
        print(f"\nSampling for scenario: {scenario}")
        filter_fn = scenario_filters.get(scenario, lambda df: df)
        filtered = filter_fn(df)
        
        # Sample clips
        sample_size = min(clips_per_scenario, len(filtered))
        sampled = filtered.sample(n=sample_size, random_state=42 + len(selected_clips))
        clip_ids = sampled["clip_id"].tolist()
        selected_clips.extend([(cid, scenario) for cid in clip_ids])
        print(f"  Selected {len(clip_ids)} clips")
    
    print(f"\nTotal clips selected: {len(selected_clips)}")
    return selected_clips


def download_clips(
    output_dir: Path,
    clip_ids: list[tuple[str, str]],
    sensors: list[str] = ["camera_front_wide_120fov", "camera_cross_left_120fov", "camera_cross_right_120fov"]
):
    """
    Download specific clips with selected sensors.
    This downloads the camera videos and ego motion labels.
    """
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    # Save clip manifest
    manifest = {
        "clips": [{"clip_id": cid, "scenario": scenario} for cid, scenario in clip_ids],
        "sensors": sensors,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")
    
    # Download using snapshot_download with allow_patterns
    # This is more efficient than individual file downloads
    
    print(f"\nDownloading {len(clip_ids)} clips...")
    print("This may take a while depending on network speed...")
    
    # For now, download the ego_motion labels which are smaller
    # Camera data is very large - we'll download a subset
    try:
        # Download ego motion data
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            local_dir=output_dir,
            allow_patterns=["labels/ego_motion/*"],
            max_workers=4,
        )
        print("Downloaded ego motion labels")
        
        # Download calibration data
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            local_dir=output_dir,
            allow_patterns=["calibration/*"],
            max_workers=4,
        )
        print("Downloaded calibration data")
        
        # Download front camera data (one chunk for testing)
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            local_dir=output_dir,
            allow_patterns=["camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip"],
            max_workers=4,
        )
        print("Downloaded camera chunk 0000")
        
    except Exception as e:
        print(f"Download error: {e}")
        import traceback
        traceback.print_exc()
    
    return clips_dir


def main():
    parser = argparse.ArgumentParser(description="Download PhysicalAI-AV dataset subset")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-clips", type=int, default=500, help="Number of clips to download")
    parser.add_argument("--scenarios", type=str, default="rain,fog,night,clear", 
                        help="Comma-separated list of scenarios")
    parser.add_argument("--list-files", action="store_true", help="Just list available files")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.list_files:
        list_available_files()
        return
    
    scenarios = [s.strip() for s in args.scenarios.split(",")]
    
    print("=" * 60)
    print("PhysicalAI-AV Dataset Download")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Target clips: {args.num_clips}")
    print(f"Scenarios: {scenarios}")
    print()
    
    # Step 1: Download metadata
    metadata_dir = download_metadata(output_dir)
    
    # Step 2: Filter clips
    clip_ids = filter_clips_by_scenario(metadata_dir, scenarios, args.num_clips)
    
    # Step 3: Download clip data
    if clip_ids:
        download_clips(output_dir, clip_ids)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
