"""
Reasoning-Action Mismatch Detector — Batch-resumable version for 2k clips.

Loads the model once, processes clips from a shared eval_clips parquet,
saves results to a single persistent CSV, and automatically resumes
from where the last job left off.

Each SLURM batch processes up to --batch-size clips (default 400, ~5.5h).
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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")
os.environ["HF_HOME"] = "/mnt/slurm_nfs/a6abdulm/.cache/huggingface"

sys.path.insert(0, str(Path(__file__).parent))
from mismatch_scorer import MismatchScorer

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "mismatch_2k"


def print_gpu(label: str = ""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM {label}] {a:.2f}GB alloc / {r:.2f}GB reserved")


class AlpamayoInference:
    def __init__(self, model_id="nvidia/Alpamayo-R1-10B"):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def load(self):
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        from alpamayo_r1 import helper

        print(f"Loading {self.model_id} (bf16, device_map=auto)...")
        print_gpu("pre-load")

        self.model = AlpamayoR1.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "18GiB", "cpu": "8GiB"},
        )
        self.processor = helper.get_processor(self.model.tokenizer)

        print_gpu("post-load")
        dm = getattr(self.model, "hf_device_map", {})
        cpu = [k for k, v in dm.items() if v == "cpu"]
        if cpu:
            print(f"  {len(cpu)} layers on CPU")

    def infer_clip(self, clip_id, t0_us, avdi):
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
        pred_np = pred_xyz.cpu().numpy()[0, 0, 0]
        gt_xyz = data["ego_future_xyz"].cpu().numpy()[0, 0]
        return cot_text, pred_np, gt_xyz

    def unload(self):
        del self.model, self.processor
        self.model = self.processor = None
        gc.collect()
        torch.cuda.empty_cache()


def load_completed_clip_ids(results_csv: Path) -> set:
    if not results_csv.exists():
        return set()
    df = pd.read_csv(results_csv)
    return set(df["clip_id"].values)


def save_results_append(results_csv: Path, new_rows: list[dict]):
    """Append rows to the persistent results CSV."""
    write_header = not results_csv.exists()
    df = pd.DataFrame(new_rows)
    df.to_csv(results_csv, mode="a", header=write_header, index=False)


def run_batch(clip_parquet: Path, batch_size: int):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = RESULTS_DIR / "all_results.csv"

    all_clips = pd.read_parquet(clip_parquet)
    done_ids = load_completed_clip_ids(results_csv)

    remaining = all_clips[~all_clips["clip_id"].isin(done_ids)]
    batch = remaining.head(batch_size)

    print("=" * 60)
    print(f"MISMATCH EXPERIMENT — batch run")
    print(f"Total clips: {len(all_clips)}")
    print(f"Already done: {len(done_ids)}")
    print(f"This batch:   {len(batch)}")
    print(f"Remaining after: {len(remaining) - len(batch)}")
    print(f"Start: {datetime.now().isoformat()}")
    print("=" * 60)

    if len(batch) == 0:
        print("Nothing to do — all clips processed!")
        return

    scorer = MismatchScorer()
    model = AlpamayoInference()
    model.load()

    import physical_ai_av
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    buffer = []
    global_idx = len(done_ids)

    for i, (_, row) in enumerate(batch.iterrows()):
        cid = row["clip_id"]
        t0 = int(row["t0_us"])
        stype = row.get("hour_bucket", "unknown")

        print(f"\n[{global_idx + i + 1}/{len(all_clips)}] clip={cid[:12]}... "
              f"t0={t0} type={stype}")

        try:
            cot_text, pred_xyz, gt_xyz = model.infer_clip(cid, t0_us=t0, avdi=avdi)
            print(f"  CoT: {cot_text[:100]}...")

            m = scorer.score(cot_text, pred_xyz)
            ade = float(np.linalg.norm(pred_xyz[:, :2] - gt_xyz[:, :2], axis=1).mean())

            result = {
                "clip_id": cid,
                "t0_us": t0,
                "scenario_type": stype,
                "cot_text": cot_text[:1000],
                "long_intent": m.longitudinal_intent.value,
                "lat_intent": m.lateral_intent.value,
                "long_execution": m.longitudinal_execution.value,
                "lat_execution": m.lateral_execution.value,
                "long_match": m.longitudinal_match,
                "lat_match": m.lateral_match,
                "mismatch_score": m.mismatch_score,
                "mismatch_type": m.mismatch_type,
                "confidence": m.text_confidence,
                "ade_meters": ade,
                "lat_shift_m": m.trajectory_features.total_lateral_shift,
                "mean_speed_ms": m.trajectory_features.mean_speed,
            }
            buffer.append(result)

            print(f"  Long: {m.longitudinal_intent.value} -> "
                  f"{m.longitudinal_execution.value} ({m.longitudinal_match:.2f})")
            print(f"  Lat:  {m.lateral_intent.value} -> "
                  f"{m.lateral_execution.value} ({m.lateral_match:.2f})")
            print(f"  Mismatch={m.mismatch_score:.3f} ({m.mismatch_type})  ADE={ade:.2f}m")

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            buffer.append({
                "clip_id": cid, "t0_us": t0, "scenario_type": stype,
                "error": str(e),
            })

        print_gpu("iter-end")

        # Flush every 10 clips
        if len(buffer) >= 10:
            save_results_append(results_csv, buffer)
            print(f"  [saved] {global_idx + i + 1}/{len(all_clips)} total")
            buffer.clear()

    # Final flush
    if buffer:
        save_results_append(results_csv, buffer)

    model.unload()

    # Print batch summary
    done_now = load_completed_clip_ids(results_csv)
    total_done = len(done_now)
    print("\n" + "=" * 60)
    print(f"BATCH COMPLETE")
    print(f"Processed this batch: {len(batch)}")
    print(f"Total done: {total_done}/{len(all_clips)}")
    print(f"Remaining: {len(all_clips) - total_done}")
    print(f"End: {datetime.now().isoformat()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip-parquet",
        type=str,
        default=str(PROJECT_ROOT / "data" / "eval_clips_2k.parquet"),
    )
    parser.add_argument("--batch-size", type=int, default=400)
    args = parser.parse_args()
    run_batch(Path(args.clip_parquet), args.batch_size)


if __name__ == "__main__":
    main()
