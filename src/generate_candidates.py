"""Generate reproducible Alpamayo trajectory candidates.

The writer is safe for a SLURM array: each clip owns one immutable record and
one compressed artifact.  No worker appends to a shared file.  A per-shard
manifest is rebuilt atomically when the worker exits.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_REVISION = "2ae73f49ffd2b5db43b404201beb7b92889f7afc"
DEFAULT_MODEL_REVISION = "69f9e9ba94445c81d8d802b048883fc473326137"
SCHEMA_VERSION = 1
TRAJECTORY_STEPS = 64
TRAJECTORY_DT_S = 0.1


@dataclass(frozen=True)
class GenerationConfig:
    model_id: str = "nvidia/Alpamayo-R1-10B"
    model_revision: str = DEFAULT_MODEL_REVISION
    dataset_revision: str = DEFAULT_DATASET_REVISION
    num_candidates: int = 6
    samples_per_rollout: int = 1
    max_generation_length: int = 256
    camera_num_frames: int = 4
    top_p: float = 0.98
    temperature: float = 0.6
    base_seed: int = 42

    def validate(self) -> None:
        if self.num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        if self.samples_per_rollout != 1:
            raise ValueError(
                "only samples_per_rollout=1 is validated for low-memory generation"
            )
        if self.max_generation_length < 1:
            raise ValueError("max_generation_length must be positive")
        if self.camera_num_frames < 1:
            raise ValueError("camera_num_frames must be positive")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateMetrics:
    ade_m: float
    fde_m: float
    mean_speed_mps: float
    mean_abs_accel_mps2: float
    mean_abs_jerk_mps3: float
    max_curvature_inv_m: float
    path_length_m: float


def log(message: str) -> None:
    print(message, flush=True)


def derive_clip_seed(base_seed: int, clip_id: str, t0_us: int) -> int:
    """Derive a stable, order-independent seed for one clip timestamp."""
    value = f"{base_seed}\0{clip_id}\0{t0_us}".encode("utf-8")
    # Torch accepts signed 64-bit seeds. Keep the high bit clear.
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") & ((1 << 63) - 1)


def derive_candidate_seed(
    base_seed: int, clip_id: str, t0_us: int, candidate_index: int
) -> int:
    """Derive one stable seed so each sequential rollout is independently replayable."""
    if candidate_index < 0:
        raise ValueError("candidate_index must be non-negative")
    value = f"{base_seed}\0{clip_id}\0{t0_us}\0{candidate_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") & ((1 << 63) - 1)


def candidate_seed_schedule(
    config: GenerationConfig, clip_id: str, t0_us: int
) -> np.ndarray:
    return np.asarray(
        [
            derive_candidate_seed(config.base_seed, clip_id, t0_us, index)
            for index in range(config.num_candidates)
        ],
        dtype=np.int64,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def artifact_key(clip_id: str, t0_us: int) -> str:
    digest = hashlib.sha256(f"{clip_id}\0{t0_us}".encode("utf-8")).hexdigest()[:20]
    return f"{digest}_{t0_us}"


def candidate_times_s() -> np.ndarray:
    """Return Alpamayo's 64 future sample times (0.1 s through 6.4 s)."""
    return np.arange(1, TRAJECTORY_STEPS + 1, dtype=np.float32) * np.float32(
        TRAJECTORY_DT_S
    )


def concatenate_candidate_rollouts(
    rollouts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Concatenate ordered, CPU-backed single-candidate rollout results."""
    if not rollouts:
        raise ValueError("at least one candidate rollout is required")
    array_fields = ("pred_xyz", "pred_rot", "candidate_seeds")
    text_fields = ("candidate_coc", "candidate_meta_action", "candidate_answer")
    combined: dict[str, Any] = {}
    for field in array_fields:
        try:
            combined[field] = np.concatenate(
                [np.asarray(rollout[field]) for rollout in rollouts], axis=0
            )
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"cannot concatenate rollout field {field}: {error}"
            ) from error
    for field in text_fields:
        try:
            combined[field] = [
                value for rollout in rollouts for value in rollout[field]
            ]
        except KeyError as error:
            raise ValueError(f"rollout is missing text field {field}") from error
    return combined


def fresh_rollout_inputs(model_inputs: dict[str, Any]) -> dict[str, Any]:
    """Copy the token mapping that upstream rollout mutates by popping input_ids."""
    tokenized_data = model_inputs.get("tokenized_data")
    if not isinstance(tokenized_data, dict):
        raise ValueError("model_inputs must contain a tokenized_data mapping")
    if "input_ids" not in tokenized_data:
        raise ValueError("tokenized_data is missing input_ids")
    return {**model_inputs, "tokenized_data": dict(tokenized_data)}


def select_shard(
    frame: pd.DataFrame, shard_index: int, num_shards: int
) -> pd.DataFrame:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    positions = np.arange(len(frame))
    return frame.iloc[positions % num_shards == shard_index].copy()


def compute_candidate_metrics(
    pred_xyz: np.ndarray, gt_xyz: np.ndarray, dt_s: float = 0.1
) -> CandidateMetrics:
    pred_xyz = np.asarray(pred_xyz, dtype=np.float64)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float64)
    if pred_xyz.ndim != 2 or pred_xyz.shape[1] != 3:
        raise ValueError(f"pred_xyz must have shape (T, 3), got {pred_xyz.shape}")
    if gt_xyz.shape != pred_xyz.shape:
        raise ValueError(f"gt_xyz shape {gt_xyz.shape} does not match {pred_xyz.shape}")
    if len(pred_xyz) < 2:
        raise ValueError("trajectories must contain at least two waypoints")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    if not np.isfinite(pred_xyz).all() or not np.isfinite(gt_xyz).all():
        raise ValueError("trajectory coordinates must be finite")

    error = np.linalg.norm(pred_xyz[:, :2] - gt_xyz[:, :2], axis=1)
    trajectory_xy = np.concatenate(
        (np.zeros((1, 2), dtype=np.float64), pred_xyz[:, :2]), axis=0
    )
    delta_xy = np.diff(trajectory_xy, axis=0)
    speed = np.linalg.norm(delta_xy, axis=1) / dt_s
    accel = np.diff(speed) / dt_s
    jerk = np.diff(accel) / dt_s

    vx = np.gradient(trajectory_xy[:, 0], dt_s)
    vy = np.gradient(trajectory_xy[:, 1], dt_s)
    ax = np.gradient(vx, dt_s)
    ay = np.gradient(vy, dt_s)
    speed_squared = vx * vx + vy * vy
    curvature = np.zeros_like(speed_squared)
    moving = speed_squared > 1e-8
    curvature[moving] = np.abs(
        vx[moving] * ay[moving] - vy[moving] * ax[moving]
    ) / np.power(speed_squared[moving], 1.5)

    metrics = CandidateMetrics(
        ade_m=float(np.mean(error)),
        fde_m=float(error[-1]),
        mean_speed_mps=float(np.mean(speed)),
        mean_abs_accel_mps2=float(np.mean(np.abs(accel))) if len(accel) else 0.0,
        mean_abs_jerk_mps3=float(np.mean(np.abs(jerk))) if len(jerk) else 0.0,
        max_curvature_inv_m=float(np.max(curvature)),
        path_length_m=float(np.sum(np.linalg.norm(delta_xy, axis=1))),
    )
    if not all(math.isfinite(value) for value in asdict(metrics).values()):
        raise ValueError("candidate metrics must be finite")
    return metrics


def validate_outputs(
    pred_xyz: np.ndarray,
    pred_rot: np.ndarray,
    gt_xyz: np.ndarray,
    gt_rot: np.ndarray,
    candidate_coc: Sequence[str],
    candidate_seeds: Sequence[int],
    expected_candidates: int,
) -> None:
    if gt_xyz.shape != (TRAJECTORY_STEPS, 3):
        raise ValueError(
            f"unexpected ground-truth xyz shape {gt_xyz.shape}; "
            f"expected ({TRAJECTORY_STEPS}, 3)"
        )
    expected_xyz = (expected_candidates, gt_xyz.shape[0], 3)
    expected_rot = (expected_candidates, gt_xyz.shape[0], 3, 3)
    if gt_rot.shape != (gt_xyz.shape[0], 3, 3):
        raise ValueError(f"unexpected ground-truth rotation shape {gt_rot.shape}")
    if pred_xyz.shape != expected_xyz:
        raise ValueError(
            f"unexpected predicted xyz shape {pred_xyz.shape}; expected {expected_xyz}"
        )
    if pred_rot.shape != expected_rot:
        raise ValueError(
            f"unexpected predicted rotation shape {pred_rot.shape}; expected {expected_rot}"
        )
    if len(candidate_coc) != expected_candidates:
        raise ValueError(
            f"received {len(candidate_coc)} CoC traces for {expected_candidates} candidates"
        )
    seeds = np.asarray(candidate_seeds)
    if seeds.shape != (expected_candidates,) or not np.issubdtype(
        seeds.dtype, np.integer
    ):
        raise ValueError(
            f"candidate_seeds must be an integer array with shape ({expected_candidates},)"
        )
    if len(np.unique(seeds)) != expected_candidates:
        raise ValueError("candidate seeds must be unique within a clip")
    arrays = (pred_xyz, pred_rot, gt_xyz, gt_rot)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("model outputs must contain only finite values")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_saved_artifact(
    path: Path,
    config: GenerationConfig,
    expected_clip_id: str,
    expected_t0_us: int,
) -> None:
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "pred_xyz",
            "pred_rot",
            "gt_xyz",
            "gt_rot",
            "candidate_coc",
            "candidate_meta_action",
            "candidate_answer",
            "config_fingerprint",
            "schema_version",
            "clip_id",
            "t0_us",
            "candidate_times_s",
            "candidate_seeds",
            "clip_seed",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"artifact {path} is missing keys: {sorted(missing)}")
        clip_id = str(saved["clip_id"].item())
        t0_us = int(saved["t0_us"].item())
        if (clip_id, t0_us) != (expected_clip_id, expected_t0_us):
            raise ValueError(
                "artifact identity "
                f"{clip_id}@{t0_us} does not match {expected_clip_id}@{expected_t0_us}"
            )
        expected_seeds = candidate_seed_schedule(
            config, expected_clip_id, expected_t0_us
        )
        validate_outputs(
            saved["pred_xyz"],
            saved["pred_rot"],
            saved["gt_xyz"],
            saved["gt_rot"],
            saved["candidate_coc"].tolist(),
            saved["candidate_seeds"],
            config.num_candidates,
        )
        if not np.array_equal(saved["candidate_seeds"], expected_seeds):
            raise ValueError(
                "artifact candidate seeds do not match the deterministic schedule"
            )
        for field in (
            "candidate_coc",
            "candidate_meta_action",
            "candidate_answer",
        ):
            if saved[field].shape != (config.num_candidates,):
                raise ValueError(f"unexpected {field} shape {saved[field].shape}")
        expected_clip_seed = derive_clip_seed(
            config.base_seed, expected_clip_id, expected_t0_us
        )
        if int(saved["clip_seed"].item()) != expected_clip_seed:
            raise ValueError("artifact clip seed does not match the deterministic schedule")
        times = np.asarray(saved["candidate_times_s"], dtype=np.float32)
        expected_times = candidate_times_s()
        if times.shape != expected_times.shape or not np.allclose(
            times, expected_times, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                "candidate_times_s must contain the 64 samples from 0.1 s to 6.4 s"
            )
        fingerprint = str(saved["config_fingerprint"].item())
        if fingerprint != config.fingerprint:
            raise ValueError(
                f"artifact configuration {fingerprint} does not match {config.fingerprint}"
            )
        if int(saved["schema_version"].item()) != SCHEMA_VERSION:
            raise ValueError(f"unsupported artifact schema in {path}")


def completion_status(
    record_path: Path,
    artifact_path: Path,
    config: GenerationConfig,
    expected_clip_id: str,
    expected_t0_us: int,
) -> tuple[bool, str | None]:
    if not record_path.exists():
        return False, None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"invalid record {record_path}: {error}"
    if record.get("schema_version") != SCHEMA_VERSION:
        return False, f"unsupported record schema in {record_path}"
    record_identity = (str(record.get("clip_id")), record.get("t0_us"))
    if record_identity != (expected_clip_id, expected_t0_us):
        return False, (
            f"record identity {record_identity[0]}@{record_identity[1]} does not match "
            f"{expected_clip_id}@{expected_t0_us}"
        )
    expected_key = artifact_key(expected_clip_id, expected_t0_us)
    expected_record_name = f"{expected_key}.json"
    expected_artifact_name = f"{expected_key}.npz"
    if (
        record_path.name != expected_record_name
        or artifact_path.name != expected_artifact_name
    ):
        return (
            False,
            "record or artifact filename does not match the clip identity hash",
        )
    expected_relative_artifact = (Path("artifacts") / expected_artifact_name).as_posix()
    if record.get("artifact_path") != expected_relative_artifact:
        return False, (
            f"record artifact_path must be {expected_relative_artifact}; "
            f"found {record.get('artifact_path')!r}"
        )
    if record.get("config_fingerprint") != config.fingerprint:
        return (
            False,
            "existing record was produced with a different generation configuration",
        )
    if record.get("config") != asdict(config):
        return False, "record configuration does not match its fingerprint"
    if not isinstance(record.get("hour_bucket"), str):
        return False, "record hour_bucket must be a string"
    expected_clip_seed = derive_clip_seed(
        config.base_seed, expected_clip_id, expected_t0_us
    )
    if record.get("clip_seed") != expected_clip_seed:
        return False, "record clip seed does not match the deterministic schedule"
    expected_seeds = candidate_seed_schedule(
        config, expected_clip_id, expected_t0_us
    ).tolist()
    if record.get("candidate_seeds") != expected_seeds:
        return False, "record candidate seeds do not match the deterministic schedule"
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != config.num_candidates:
        return False, "record candidates do not match the configured candidate count"
    for index, (candidate, seed) in enumerate(zip(candidates, expected_seeds)):
        if not isinstance(candidate, dict) or (
            candidate.get("candidate_index"),
            candidate.get("candidate_seed"),
        ) != (index, seed):
            return False, f"record candidate {index} has an invalid index or seed"
    if not artifact_path.exists():
        return False, f"record exists but artifact is missing: {artifact_path}"
    try:
        actual_sha256 = file_sha256(artifact_path)
        if record.get("artifact_sha256") != actual_sha256:
            return False, "record artifact_sha256 does not match the artifact contents"
        validate_saved_artifact(artifact_path, config, expected_clip_id, expected_t0_us)
        with np.load(artifact_path, allow_pickle=False) as saved:
            expected_metrics = [
                asdict(compute_candidate_metrics(candidate, saved["gt_xyz"]))
                for candidate in saved["pred_xyz"]
            ]
            expected_oracle_index = int(
                np.argmin([metrics["ade_m"] for metrics in expected_metrics])
            )
            if record.get("oracle_min_ade_candidate_index") != expected_oracle_index:
                return False, "record oracle candidate does not match the artifact"
            text_fields = {
                "coc": saved["candidate_coc"].tolist(),
                "meta_action": saved["candidate_meta_action"].tolist(),
                "answer": saved["candidate_answer"].tolist(),
            }
            for index, candidate in enumerate(candidates):
                if candidate.get("metrics") != expected_metrics[index]:
                    return False, f"record candidate {index} metrics do not match the artifact"
                if candidate.get("is_oracle_min_ade") != (
                    index == expected_oracle_index
                ):
                    return False, f"record candidate {index} oracle flag is invalid"
                for record_field, expected_values in text_fields.items():
                    if candidate.get(record_field) != expected_values[index]:
                        return False, (
                            f"record candidate {index} {record_field} does not match "
                            "the artifact"
                        )
    except (OSError, TypeError, ValueError) as error:
        return False, f"invalid artifact {artifact_path}: {error}"
    return True, None


def _candidate_array(
    tensor: torch.Tensor, expected_candidates: int, name: str
) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.shape[:3] != (1, 1, expected_candidates):
        raise ValueError(f"unexpected {name} leading shape {array.shape}")
    return array[0, 0]


def _ground_truth_array(tensor: torch.Tensor, name: str) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.shape[:2] != (1, 1):
        raise ValueError(f"unexpected {name} leading shape {array.shape}")
    return array[0, 0]


def _candidate_text(
    extra: dict[str, Any], field: str, expected_candidates: int
) -> list[str]:
    values = np.asarray(extra.get(field, []), dtype=str)
    if values.shape != (1, 1, expected_candidates):
        raise ValueError(f"unexpected {field} text shape {values.shape}")
    return values[0, 0].tolist()


class AlpamayoCandidateGenerator:
    def __init__(self, config: GenerationConfig, gpu_memory: str, cpu_memory: str):
        self.config = config
        self.gpu_memory = gpu_memory
        self.cpu_memory = cpu_memory
        self.model: Any = None
        self.processor: Any = None

    def load(self) -> None:
        from alpamayo_r1 import helper
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

        visible_gpus = torch.cuda.device_count()
        if visible_gpus != 1:
            raise RuntimeError(
                "candidate generation requires exactly one visible GPU; request one GPU "
                "from the scheduler or restrict CUDA_VISIBLE_DEVICES"
            )
        log(
            f"Loading {self.config.model_id}@{self.config.model_revision} "
            "in bfloat16 with one-GPU/CPU automatic placement"
        )
        self.model = AlpamayoR1.from_pretrained(
            self.config.model_id,
            revision=self.config.model_revision,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: self.gpu_memory, "cpu": self.cpu_memory},
        )
        self.processor = helper.get_processor(self.model.tokenizer)

    def unload(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def infer(self, clip_id: str, t0_us: int, avdi: Any) -> dict[str, Any]:
        from alpamayo_r1 import helper
        from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

        data = load_physical_aiavdataset(
            clip_id,
            t0_us=t0_us,
            avdi=avdi,
            num_frames=self.config.camera_num_frames,
        )
        messages = helper.create_message(data["image_frames"].flatten(0, 1))
        tokenized = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = helper.to_device(
            {
                "tokenized_data": tokenized,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            },
            "cuda",
        )

        candidate_seeds = candidate_seed_schedule(self.config, clip_id, t0_us)
        rollouts: list[dict[str, Any]] = []
        for candidate_index, candidate_seed in enumerate(candidate_seeds):
            log(
                f"  rollout {candidate_index + 1}/{self.config.num_candidates} "
                f"seed={int(candidate_seed)}"
            )
            seed_everything(int(candidate_seed))
            rollout_inputs = fresh_rollout_inputs(model_inputs)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = (
                    self.model.sample_trajectories_from_data_with_vlm_rollout(
                        data=rollout_inputs,
                        top_p=self.config.top_p,
                        temperature=self.config.temperature,
                        num_traj_samples=self.config.samples_per_rollout,
                        max_generation_length=self.config.max_generation_length,
                        return_extra=True,
                    )
                )
            rollouts.append(
                {
                    "pred_xyz": _candidate_array(pred_xyz, 1, "pred_xyz"),
                    "pred_rot": _candidate_array(pred_rot, 1, "pred_rot"),
                    "candidate_coc": _candidate_text(extra, "cot", 1),
                    "candidate_meta_action": _candidate_text(extra, "meta_action", 1),
                    "candidate_answer": _candidate_text(extra, "answer", 1),
                    "candidate_seeds": np.asarray([candidate_seed], dtype=np.int64),
                }
            )
            del pred_xyz, pred_rot, extra, rollout_inputs
            gc.collect()
            torch.cuda.empty_cache()

        candidate_output = concatenate_candidate_rollouts(rollouts)
        count = self.config.num_candidates
        output = {
            **candidate_output,
            "gt_xyz": _ground_truth_array(data["ego_future_xyz"], "ego_future_xyz"),
            "gt_rot": _ground_truth_array(data["ego_future_rot"], "ego_future_rot"),
        }
        validate_outputs(
            output["pred_xyz"],
            output["pred_rot"],
            output["gt_xyz"],
            output["gt_rot"],
            output["candidate_coc"],
            output["candidate_seeds"],
            count,
        )
        return output


def _load_clips(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"clip_id", "t0_us"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"clip parquet is missing columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("clip_id and t0_us must not contain null values")
    duplicate = frame.duplicated(["clip_id", "t0_us"])
    if duplicate.any():
        raise ValueError(
            f"clip parquet contains {int(duplicate.sum())} duplicate clip timestamps"
        )
    return frame.reset_index(drop=True)


def _record_for_output(
    clip_id: str,
    t0_us: int,
    hour_bucket: str,
    artifact_path: Path,
    artifact_sha256: str,
    output_dir: Path,
    output: dict[str, Any],
    config: GenerationConfig,
    clip_seed: int,
    elapsed_s: float,
) -> dict[str, Any]:
    metrics = [
        asdict(compute_candidate_metrics(candidate, output["gt_xyz"]))
        for candidate in output["pred_xyz"]
    ]
    oracle_index = int(np.argmin([item["ade_m"] for item in metrics]))
    candidates = []
    for index, metric in enumerate(metrics):
        candidate_seed = int(output["candidate_seeds"][index])
        candidates.append(
            {
                "candidate_index": index,
                "candidate_seed": candidate_seed,
                "coc": output["candidate_coc"][index],
                "meta_action": output["candidate_meta_action"][index],
                "answer": output["candidate_answer"][index],
                "metrics": metric,
                "is_oracle_min_ade": index == oracle_index,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        "t0_us": t0_us,
        "hour_bucket": hour_bucket,
        "clip_seed": clip_seed,
        "candidate_seeds": [int(seed) for seed in output["candidate_seeds"]],
        "config": asdict(config),
        "config_fingerprint": config.fingerprint,
        "artifact_path": str(artifact_path.relative_to(output_dir)),
        "artifact_sha256": artifact_sha256,
        "oracle_min_ade_candidate_index": oracle_index,
        "candidates": candidates,
        "elapsed_s": elapsed_s,
        "runtime": runtime_metadata(),
    }


def runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        metadata["gpu"] = torch.cuda.get_device_name(0)
        metadata["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    return metadata


def _write_clip(
    record_path: Path,
    artifact_path: Path,
    output_dir: Path,
    clip_id: str,
    t0_us: int,
    hour_bucket: str,
    output: dict[str, Any],
    config: GenerationConfig,
    clip_seed: int,
    elapsed_s: float,
) -> dict[str, Any]:
    atomic_write_npz(
        artifact_path,
        pred_xyz=np.asarray(output["pred_xyz"], dtype=np.float32),
        pred_rot=np.asarray(output["pred_rot"], dtype=np.float32),
        gt_xyz=np.asarray(output["gt_xyz"], dtype=np.float32),
        gt_rot=np.asarray(output["gt_rot"], dtype=np.float32),
        candidate_coc=np.asarray(output["candidate_coc"], dtype=np.str_),
        candidate_meta_action=np.asarray(
            output["candidate_meta_action"], dtype=np.str_
        ),
        candidate_answer=np.asarray(output["candidate_answer"], dtype=np.str_),
        candidate_seeds=np.asarray(output["candidate_seeds"], dtype=np.int64),
        candidate_times_s=candidate_times_s(),
        config_fingerprint=np.asarray(config.fingerprint, dtype=np.str_),
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
        clip_id=np.asarray(clip_id, dtype=np.str_),
        t0_us=np.asarray(t0_us, dtype=np.int64),
        clip_seed=np.asarray(clip_seed, dtype=np.int64),
    )
    validate_saved_artifact(artifact_path, config, clip_id, t0_us)
    artifact_sha256 = file_sha256(artifact_path)
    with np.load(artifact_path, allow_pickle=False) as saved:
        persisted_output = {
            "pred_xyz": saved["pred_xyz"].copy(),
            "gt_xyz": saved["gt_xyz"].copy(),
            "candidate_coc": saved["candidate_coc"].tolist(),
            "candidate_meta_action": saved["candidate_meta_action"].tolist(),
            "candidate_answer": saved["candidate_answer"].tolist(),
            "candidate_seeds": saved["candidate_seeds"].copy(),
        }
    record = _record_for_output(
        clip_id,
        t0_us,
        hour_bucket,
        artifact_path,
        artifact_sha256,
        output_dir,
        persisted_output,
        config,
        clip_seed,
        elapsed_s,
    )
    atomic_write_json(record_path, record)
    return record


def run_generation(args: argparse.Namespace) -> int:
    config = GenerationConfig(
        model_id=args.model_id,
        model_revision=args.model_revision,
        dataset_revision=args.dataset_revision,
        num_candidates=args.num_candidates,
        max_generation_length=args.max_generation_length,
        camera_num_frames=args.camera_num_frames,
        top_p=args.top_p,
        temperature=args.temperature,
        base_seed=args.base_seed,
    )
    config.validate()
    if args.max_clips is not None and args.max_clips < 1:
        raise ValueError("max_clips must be positive when provided")
    output_dir = args.output_dir.resolve()
    artifact_dir = output_dir / "artifacts"
    record_dir = output_dir / "records"
    failure_dir = output_dir / "failures"
    for directory in (artifact_dir, record_dir, failure_dir, output_dir / "manifests"):
        directory.mkdir(parents=True, exist_ok=True)

    clips = select_shard(
        _load_clips(args.clip_parquet), args.shard_index, args.num_shards
    )
    if args.max_clips is not None:
        clips = clips.head(args.max_clips)
    log(
        f"Shard {args.shard_index}/{args.num_shards}: {len(clips)} clips, "
        f"K={config.num_candidates}, config={config.fingerprint[:12]}"
    )

    pending: list[tuple[pd.Series, str, Path, Path, Path]] = []
    manifest_records: list[dict[str, Any]] = []
    for _, row in clips.iterrows():
        clip_id, t0_us = str(row["clip_id"]), int(row["t0_us"])
        key = artifact_key(clip_id, t0_us)
        record_path = record_dir / f"{key}.json"
        artifact_path = artifact_dir / f"{key}.npz"
        failure_path = failure_dir / f"{key}.json"
        complete, problem = completion_status(
            record_path,
            artifact_path,
            config,
            clip_id,
            t0_us,
        )
        if complete and not args.overwrite:
            manifest_records.append(json.loads(record_path.read_text(encoding="utf-8")))
            continue
        if record_path.exists() and not args.overwrite:
            raise RuntimeError(
                f"cannot resume {clip_id}@{t0_us}: {problem}; use --overwrite"
            )
        pending.append((row, key, record_path, artifact_path, failure_path))

    log(f"Completed: {len(manifest_records)}; pending: {len(pending)}")
    failures = 0
    generator: AlpamayoCandidateGenerator | None = None
    if pending:
        visible_gpus = torch.cuda.device_count()
        if visible_gpus != 1:
            raise RuntimeError(
                "candidate generation requires exactly one visible GPU; request one GPU "
                "from the scheduler or restrict CUDA_VISIBLE_DEVICES"
            )
        import physical_ai_av

        avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
            revision=config.dataset_revision
        )
        generator = AlpamayoCandidateGenerator(config, args.gpu_memory, args.cpu_memory)
        generator.load()
        try:
            for index, (
                row,
                key,
                record_path,
                artifact_path,
                failure_path,
            ) in enumerate(pending, start=1):
                clip_id, t0_us = str(row["clip_id"]), int(row["t0_us"])
                hour_bucket = str(row.get("hour_bucket", "unknown"))
                clip_seed = derive_clip_seed(config.base_seed, clip_id, t0_us)
                log(f"[{index}/{len(pending)}] {clip_id}@{t0_us} seed={clip_seed}")
                started = time.monotonic()
                try:
                    output = generator.infer(clip_id, t0_us, avdi)
                    record = _write_clip(
                        record_path,
                        artifact_path,
                        output_dir,
                        clip_id,
                        t0_us,
                        hour_bucket,
                        output,
                        config,
                        clip_seed,
                        time.monotonic() - started,
                    )
                    manifest_records.append(record)
                    failure_path.unlink(missing_ok=True)
                    oracle = record["oracle_min_ade_candidate_index"]
                    oracle_ade = record["candidates"][oracle]["metrics"]["ade_m"]
                    log(f"  wrote {key}; oracle minADE={oracle_ade:.3f} m")
                except (
                    Exception
                ) as error:  # preserve later clips while recording failure context
                    failures += 1
                    atomic_write_json(
                        failure_path,
                        {
                            "clip_id": clip_id,
                            "t0_us": t0_us,
                            "candidate_seeds": candidate_seed_schedule(
                                config, clip_id, t0_us
                            ).tolist(),
                            "config_fingerprint": config.fingerprint,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    log(f"  ERROR: {type(error).__name__}: {error}")
                    if args.fail_fast:
                        raise
        finally:
            generator.unload()

    manifest_records.sort(key=lambda item: (item["clip_id"], item["t0_us"]))
    manifest_path = (
        output_dir
        / "manifests"
        / (f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}.jsonl")
    )
    atomic_write_jsonl(manifest_path, manifest_records)
    log(f"Manifest: {manifest_path} ({len(manifest_records)} completed clips)")
    if failures:
        log(f"Failed clips: {failures}; details are in {failure_dir}")
        return 1
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip-parquet",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval_clips_2k.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "candidates_k6",
    )
    parser.add_argument("--model-id", default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--camera-num-frames", type=int, default=4)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--gpu-memory", default="18GiB")
    parser.add_argument("--cpu-memory", default="8GiB")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_generation(parse_args(argv))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
