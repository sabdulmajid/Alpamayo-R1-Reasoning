"""
Reasoning-Action Mismatch Detector for Alpamayo-R1-10B.

Runs the model on PhysicalAI-AV clips and quantifies how often
the Chain-of-Causation text reasoning disagrees with the predicted trajectory.
"""

import argparse
import gc
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# NVML workaround — must be set before any CUDA calls
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")
os.environ["HF_HOME"] = "/mnt/slurm_nfs/a6abdulm/.cache/huggingface"

from mismatch_scorer import MismatchScorer

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def print_gpu_memory(label: str = ""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        res = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM {label}] Allocated: {alloc:.2f}GB | Reserved: {res:.2f}GB")


class AlpamayoInference:
    """Wrapper for Alpamayo-R1 model using the official alpamayo_r1 API."""

    def __init__(self, model_id: str = "nvidia/Alpamayo-R1-10B"):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def load(self):
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        from alpamayo_r1 import helper

        print(f"Loading {self.model_id} with device_map='auto' (bf16)...")
        print_gpu_memory("pre-load")

        self.model = AlpamayoR1.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "18GiB", "cpu": "8GiB"},
        )
        self.processor = helper.get_processor(self.model.tokenizer)

        print_gpu_memory("post-load")
        device_map = getattr(self.model, "hf_device_map", {})
        cpu_layers = [k for k, v in device_map.items() if v == "cpu"]
        if cpu_layers:
            print(f"  {len(cpu_layers)} layers offloaded to CPU")

    def infer_clip(self, clip_id: str, t0_us: int = 5_100_000, avdi=None):
        """
        Run full VLA inference on a dataset clip.

        Returns:
            cot_text: Chain-of-Causation reasoning string
            pred_xyz: predicted trajectory (num_future_steps, 3)
            gt_xyz:   ground-truth trajectory (num_future_steps, 3)
        """
        from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
        from alpamayo_r1 import helper

        data = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)
        messages = helper.create_message(data["image_frames"].flatten(0, 1))

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, "cuda")

        torch.cuda.manual_seed_all(42)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = (
                self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=256,
                    return_extra=True,
                )
            )

        cot_text = extra["cot"][0, 0, 0]
        pred_np = pred_xyz.cpu().numpy()[0, 0, 0]  # (num_future_steps, 3)
        gt_xyz = data["ego_future_xyz"].cpu().numpy()[0, 0]

        return cot_text, pred_np, gt_xyz

    def unload(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()


def get_clip_list() -> list[dict]:
    """
    Return clip_ids to evaluate.
    Checks: eval_clips.parquet > vla_golden.parquet > single example clip.
    """
    for parquet_name in ["eval_clips.parquet", "vla_golden.parquet"]:
        path = PROJECT_ROOT / "data" / parquet_name
        if path.exists():
            df = pd.read_parquet(path)
            clips = []
            for _, row in df.iterrows():
                meta = {
                    "clip_id": row["clip_id"],
                    "t0_us": int(row.get("t0_us", 5_100_000)),
                }
                if "hour_bucket" in row:
                    meta["scenario_type"] = row["hour_bucket"]
                elif "scenario_type" in row:
                    meta["scenario_type"] = row["scenario_type"]
                else:
                    meta["scenario_type"] = "unknown"
                clips.append(meta)
            print(f"Loaded {len(clips)} clips from {parquet_name}")
            return clips

    print("No eval set found — using single example clip")
    return [
        {
            "clip_id": "030c760c-ae38-49aa-9ad8-f5650a545d26",
            "t0_us": 5_100_000,
            "scenario_type": "example",
        }
    ]


def run_experiment(output_dir: Path, max_clips: int | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("REASONING-ACTION MISMATCH EXPERIMENT")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    clips = get_clip_list()
    if max_clips:
        clips = clips[:max_clips]
    print(f"Will process {len(clips)} clips")

    scorer = MismatchScorer()
    model = AlpamayoInference()
    model.load()

    # Shared dataset interface — avoids re-init per clip
    import physical_ai_av
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    results = []
    for i, clip_info in enumerate(clips):
        cid = clip_info["clip_id"]
        t0 = clip_info["t0_us"]
        stype = clip_info.get("scenario_type", "unknown")
        print(f"\n[{i+1}/{len(clips)}] clip={cid[:12]}... t0={t0} type={stype}")

        try:
            cot_text, pred_xyz, gt_xyz = model.infer_clip(cid, t0_us=t0, avdi=avdi)
            print(f"  CoT: {cot_text[:120]}...")

            mismatch = scorer.score(cot_text, pred_xyz)

            ade = float(np.linalg.norm(pred_xyz[:, :2] - gt_xyz[:, :2], axis=1).mean())

            result = {
                "clip_id": cid,
                "t0_us": t0,
                "scenario_type": stype,
                "cot_text": cot_text[:1000],
                "long_intent": mismatch.longitudinal_intent.value,
                "lat_intent": mismatch.lateral_intent.value,
                "long_execution": mismatch.longitudinal_execution.value,
                "lat_execution": mismatch.lateral_execution.value,
                "long_match": mismatch.longitudinal_match,
                "lat_match": mismatch.lateral_match,
                "mismatch_score": mismatch.mismatch_score,
                "mismatch_type": mismatch.mismatch_type,
                "confidence": mismatch.text_confidence,
                "ade_meters": ade,
                "lat_shift_m": mismatch.trajectory_features.total_lateral_shift,
                "mean_speed_ms": mismatch.trajectory_features.mean_speed,
            }
            results.append(result)

            print(f"  Long: {mismatch.longitudinal_intent.value} -> "
                  f"{mismatch.longitudinal_execution.value} ({mismatch.longitudinal_match:.2f})")
            print(f"  Lat:  {mismatch.lateral_intent.value} -> "
                  f"{mismatch.lateral_execution.value} ({mismatch.lateral_match:.2f})")
            print(f"  Mismatch={mismatch.mismatch_score:.3f} ({mismatch.mismatch_type})  "
                  f"ADE={ade:.2f}m")

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({"clip_id": cid, "t0_us": t0, "error": str(e)})

        print_gpu_memory("iter-end")

        # Incremental save every 10 clips (protection against timeout)
        if (i + 1) % 10 == 0:
            pd.DataFrame(results).to_csv(
                output_dir / "partial_results.csv", index=False
            )
            print(f"  [checkpoint] {i+1}/{len(clips)} saved")

    df = pd.DataFrame(results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"mismatch_results_{ts}.csv"
    df.to_csv(csv_path, index=False)

    valid = df[df["mismatch_score"].notna()] if "mismatch_score" in df.columns else df.iloc[0:0]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Clips processed: {len(df)}")
    print(f"Valid results: {len(valid)}")
    if len(valid) > 0:
        print(f"Mean mismatch score: {valid['mismatch_score'].mean():.3f}")
        print(f"Mean ADE: {valid['ade_meters'].mean():.2f} m")

        print("\nMismatch type distribution:")
        for mtype, count in valid["mismatch_type"].value_counts().items():
            print(f"  {mtype}: {count} ({100*count/len(valid):.1f}%)")

        print("\nLongitudinal axis:")
        for li in valid["long_intent"].value_counts().items():
            print(f"  intent={li[0]}: {li[1]}")
        for le in valid["long_execution"].value_counts().items():
            print(f"  exec={le[0]}: {le[1]}")
        if "long_match" in valid.columns:
            print(f"  mean longitudinal match: {valid['long_match'].mean():.3f}")

        print("\nLateral axis:")
        for li in valid["lat_intent"].value_counts().items():
            print(f"  intent={li[0]}: {li[1]}")
        for le in valid["lat_execution"].value_counts().items():
            print(f"  exec={le[0]}: {le[1]}")
        if "lat_match" in valid.columns:
            print(f"  mean lateral match: {valid['lat_match'].mean():.3f}")

        if "scenario_type" in valid.columns:
            print("\nBy scenario type:")
            for stype in valid["scenario_type"].unique():
                sub = valid[valid["scenario_type"] == stype]
                print(f"  {stype}: mismatch={sub['mismatch_score'].mean():.3f}  "
                      f"ADE={sub['ade_meters'].mean():.2f}m  n={len(sub)}")

    summary = {
        "timestamp": ts,
        "total_clips": len(df),
        "valid": len(valid),
        "mean_mismatch": float(valid["mismatch_score"].mean()) if len(valid) else None,
        "mean_ade": float(valid["ade_meters"].mean()) if len(valid) else None,
        "mean_long_match": float(valid["long_match"].mean()) if len(valid) and "long_match" in valid.columns else None,
        "mean_lat_match": float(valid["lat_match"].mean()) if len(valid) and "lat_match" in valid.columns else None,
    }
    with open(output_dir / f"summary_{ts}.json", "w") as f:
        json.dump(summary, f, indent=2)

    model.unload()
    print(f"\nDone: {datetime.now().isoformat()}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--max-clips", type=int, default=None)
    args = parser.parse_args()
    run_experiment(Path(args.output_dir), max_clips=args.max_clips)


if __name__ == "__main__":
    main()
