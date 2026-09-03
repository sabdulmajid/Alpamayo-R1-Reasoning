import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.world_model.benchmark_prep import (
    freeze_splits,
    freeze_benchmark_protocol,
    merge_oracle_manifests,
    select_checkpoint,
    verify_evaluation_partition,
)
from src.world_model.runtime import sha256_file
from src.world_model.split_manifest import SPLIT_NAMES, assign_group


HORIZONS = [1.0, 2.0]
HISTORY = [-1.0, -0.5, 0.0]
CANDIDATE_CONFIG_PAYLOAD = {"num_candidates": 2, "seed": 42}
CANDIDATE_CONFIG = hashlib.sha256(
    json.dumps(CANDIDATE_CONFIG_PAYLOAD, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
ORACLE_CONFIG = "b" * 64
DATASET_REVISION = "c" * 40


def identity_digest(clip_id: str, t0_us: int) -> str:
    return hashlib.sha256(f"{clip_id}\0{t0_us}".encode()).hexdigest()[:20]


def write_candidate(candidate_root: Path, clip_id: str, t0_us: int, count: int) -> str:
    key = f"{identity_digest(clip_id, t0_us)}_{t0_us}"
    artifact = candidate_root / "artifacts" / f"{key}.npz"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    candidate_xyz = np.zeros((count, 3, 3), dtype=np.float32)
    candidate_xyz[:, :, 0] = np.arange(3, dtype=np.float32)
    np.savez_compressed(
        artifact,
        clip_id=np.asarray(clip_id),
        t0_us=np.int64(t0_us),
        config_fingerprint=np.asarray(CANDIDATE_CONFIG),
        pred_xyz=candidate_xyz,
    )
    artifact_sha256 = sha256_file(artifact)
    record = {
        "schema_version": 1,
        "clip_id": clip_id,
        "t0_us": t0_us,
        "artifact_path": f"artifacts/{key}.npz",
        "artifact_sha256": artifact_sha256,
        "config": CANDIDATE_CONFIG_PAYLOAD,
        "config_fingerprint": CANDIDATE_CONFIG,
        "candidates": [{"candidate_index": index} for index in range(count)],
    }
    record_dir = candidate_root / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / f"{key}.json").write_text(
        json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact_sha256


def write_oracle(
    oracle_root: Path,
    clip_id: str,
    t0_us: int,
    candidate_sha256: str,
    count: int,
) -> Path:
    artifact = oracle_root / "clips" / f"{identity_digest(clip_id, t0_us)}.oracle.npz"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    past = np.zeros((3, 8, 10), dtype=np.uint8)
    future = np.zeros((2, 8, 10), dtype=np.uint8)
    candidates = np.zeros((count, 3, 3), dtype=np.float32)
    candidates[:, :, 0] = np.arange(3, dtype=np.float32)
    np.savez_compressed(
        artifact,
        schema_version=np.int16(4),
        clip_id=np.asarray(clip_id),
        t0_us=np.int64(t0_us),
        dataset_revision=np.asarray(DATASET_REVISION),
        frame=np.asarray("ego_at_t0"),
        oracle_config_fingerprint=np.asarray(ORACLE_CONFIG),
        candidate_config_fingerprint=np.asarray(CANDIDATE_CONFIG),
        candidate_artifact_sha256=np.asarray(candidate_sha256),
        past_occupancy=past,
        past_observed=np.ones_like(past),
        occupancy=future,
        observed=np.ones_like(future),
        history_offsets_s=np.asarray(HISTORY, dtype=np.float32),
        horizons_s=np.asarray(HORIZONS, dtype=np.float32),
        candidate_xyz=candidates,
        oracle_safest_idx=np.int32(1),
        bev_x_min_m=np.float32(-2.0),
        bev_x_max_m=np.float32(3.0),
        bev_y_min_m=np.float32(-3.0),
        bev_y_max_m=np.float32(1.0),
        bev_resolution_m=np.float32(0.5),
    )
    return artifact


def split_covering_chunks(seed: int, ratios: tuple[float, float, float]) -> list[int]:
    found: dict[str, int] = {}
    for chunk_id in range(1, 10000):
        split = assign_group(str(chunk_id), seed, ratios)
        found.setdefault(split, chunk_id)
        if len(found) == len(SPLIT_NAMES):
            return [found[name] for name in SPLIT_NAMES]
    raise AssertionError("Could not find one deterministic chunk for each split")


def make_benchmark_tree(
    root: Path, *, shards: int = 2, count: int = 2
) -> tuple[Path, Path]:
    candidate_root = root / "candidates"
    oracle_root = root / "oracle"
    manifest_dir = oracle_root / "manifests"
    manifest_dir.mkdir(parents=True)
    rows_by_shard: dict[int, list[dict[str, object]]] = {
        index: [] for index in range(shards)
    }
    chunks = split_covering_chunks(2026, (0.8, 0.1, 0.1))
    for index, chunk_id in enumerate(chunks):
        clip_id = f"clip-{index}"
        t0_us = 1000 + index
        candidate_sha256 = write_candidate(candidate_root, clip_id, t0_us, count)
        artifact = write_oracle(oracle_root, clip_id, t0_us, candidate_sha256, count)
        rows_by_shard[chunk_id % shards].append(
            {
                "artifact_path": f"../clips/{artifact.name}",
                "candidate_artifact_sha256": candidate_sha256,
                "candidate_count": count,
                "chunk_id": chunk_id,
                "clip_id": clip_id,
                "dataset_revision": DATASET_REVISION,
                "horizons_s": HORIZONS,
                "history_offsets_s": HISTORY,
                "identity_digest": identity_digest(clip_id, t0_us),
                "oracle_config_fingerprint": ORACLE_CONFIG,
                "oracle_safest_idx": 1,
                "resumed": False,
                "t0_us": t0_us,
                "yaw_source": "candidate_rotation",
            }
        )
    for shard, rows in rows_by_shard.items():
        manifest = manifest_dir / f"shard-{shard:05d}-of-{shards:05d}.jsonl"
        manifest.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    return manifest_dir, candidate_root


class ManifestPreparationTest(unittest.TestCase):
    def test_merge_checks_sources_and_split_freezes_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_dir, candidate_dir = make_benchmark_tree(root)
            rows, merge_summary = merge_oracle_manifests(
                manifest_dir=manifest_dir,
                candidate_dir=candidate_dir,
                expected_shards=2,
                expected_rows=3,
                expected_candidates=2,
            )
            self.assertEqual(len(rows), 3)
            self.assertEqual(merge_summary["clips"], 3)
            self.assertEqual(len(merge_summary["dataset_fingerprint"]), 64)
            self.assertTrue(
                all(Path(row["artifact_path"]).is_absolute() for row in rows)
            )
            self.assertTrue(all(len(row["artifact_sha256"]) == 64 for row in rows))

            splits, split_summary = freeze_splits(
                rows, seed=2026, ratios=(0.8, 0.1, 0.1)
            )
            self.assertEqual([len(splits[name]) for name in SPLIT_NAMES], [1, 1, 1])
            chunk_sets = [
                {row["chunk_id"] for row in splits[name]} for name in SPLIT_NAMES
            ]
            self.assertFalse(chunk_sets[0] & chunk_sets[1])
            self.assertFalse(chunk_sets[0] & chunk_sets[2])
            self.assertFalse(chunk_sets[1] & chunk_sets[2])
            self.assertEqual(len(split_summary["split_plan_fingerprint"]), 64)
            for name in SPLIT_NAMES:
                self.assertEqual(
                    len(split_summary["splits"][name]["manifest_sha256"]), 64
                )

            split_dir = root / "splits"
            split_dir.mkdir()
            for name in SPLIT_NAMES:
                (split_dir / f"{name}.jsonl").write_text(
                    "".join(
                        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                        for row in splits[name]
                    ),
                    encoding="utf-8",
                )
            merged = root / "all.jsonl"
            merged.write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            split_summary["source_manifest"] = str(merged)
            split_summary["source_file_sha256"] = sha256_file(merged)
            split_summary["output_directory"] = str(split_dir)
            split_summary_path = root / "split_summary.json"
            split_summary_path.write_text(json.dumps(split_summary), encoding="utf-8")
            source_parquet = root / "clips.parquet"
            pd.DataFrame(
                {
                    "clip_id": [row["clip_id"] for row in rows],
                    "t0_us": [row["t0_us"] for row in rows],
                }
            ).to_parquet(source_parquet, index=False)
            protocol = freeze_benchmark_protocol(
                split_summary_path=split_summary_path,
                source_parquet=source_parquet,
                expected_source_parquet_sha256=sha256_file(source_parquet),
            )
            self.assertEqual(protocol["status"], "frozen_before_test_evaluation")
            self.assertEqual(protocol["training"]["seeds"], [2026, 2027])
            self.assertEqual(
                protocol["trajectory_selection"]["primary_outcome"],
                "collision_exposure",
            )
            self.assertEqual(len(protocol["protocol_fingerprint"]), 64)

            changed_parquet = root / "changed-clips.parquet"
            pd.DataFrame(
                {"clip_id": ["different-clip"], "t0_us": [999]}
            ).to_parquet(changed_parquet, index=False)
            with self.assertRaisesRegex(ValueError, "contain different clips"):
                freeze_benchmark_protocol(
                    split_summary_path=split_summary_path,
                    source_parquet=changed_parquet,
                    expected_source_parquet_sha256=sha256_file(changed_parquet),
                )

    def test_merge_rejects_candidate_source_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_dir, candidate_dir = make_benchmark_tree(root)
            manifest = next(
                path
                for path in sorted(manifest_dir.glob("*.jsonl"))
                if path.stat().st_size
            )
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            rows[0]["candidate_artifact_sha256"] = "0" * 64
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "candidate SHA-256 differs"):
                merge_oracle_manifests(
                    manifest_dir=manifest_dir,
                    candidate_dir=candidate_dir,
                    expected_shards=2,
                    expected_rows=3,
                    expected_candidates=2,
                )

    def test_merge_rejects_incomplete_shard_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_dir, candidate_dir = make_benchmark_tree(root)
            (manifest_dir / "shard-00001-of-00002.jsonl").rename(
                manifest_dir / "shard-00001-of-00003.jsonl"
            )
            with self.assertRaisesRegex(ValueError, "shard set is not exact"):
                merge_oracle_manifests(
                    manifest_dir=manifest_dir,
                    candidate_dir=candidate_dir,
                    expected_shards=2,
                    expected_rows=3,
                    expected_candidates=2,
                )


def write_checkpoint(run_dir: Path, *, seed: int, best_loss: float) -> None:
    run_dir.mkdir(parents=True)
    resume_signature = {
        "train_manifest": "/benchmark/train.jsonl",
        "train_manifest_sha256": "d" * 64,
        "train_dataset_fingerprint": "e" * 64,
        "val_manifest": "/benchmark/val.jsonl",
        "val_manifest_sha256": "f" * 64,
        "val_dataset_fingerprint": "1" * 64,
        "batch_size": 4,
        "base_channels": 16,
        "seed": seed,
    }
    torch.save(
        {
            "checkpoint_schema_version": 2,
            "epoch": 4,
            "best_val_loss": best_loss,
            "seed": seed,
            "history": [{"epoch": 4, "val": {"loss": best_loss}}],
            "resume_signature": resume_signature,
            "data_provenance": {
                "train": {
                    "dataset_fingerprint": "e" * 64,
                    "chunk_ids": ["chunk-train"],
                },
                "val": {
                    "dataset_fingerprint": "1" * 64,
                    "chunk_ids": ["chunk-val"],
                },
            },
            "model_config": {"base_channels": 16},
            "data_schema": {"coordinate_frame": "ego_at_t0"},
        },
        run_dir / "best.pt",
    )


class CheckpointSelectionTest(unittest.TestCase):
    def test_selection_uses_validation_loss_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_checkpoint(root / "seed-2026", seed=2026, best_loss=0.4)
            write_checkpoint(root / "seed-2027", seed=2027, best_loss=0.3)
            result = select_checkpoint(
                (("seed-2026", root / "seed-2026"), ("seed-2027", root / "seed-2027"))
            )
            self.assertEqual(result["selected_seed"], 2027)
            self.assertEqual(result["selection_policy"], "minimum_best_validation_loss")
            self.assertFalse(result["test_metrics_used"])
            self.assertEqual(len(result["selected_checkpoint_sha256"]), 64)

            result_path = root / "selection.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            test_manifest = root / "test.jsonl"
            test_manifest.write_text(
                json.dumps(
                    {
                        "clip_id": "clip-test",
                        "t0_us": 90,
                        "chunk_id": "chunk-test",
                        "artifact_path": str(root / "unused.npz"),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            verification = verify_evaluation_partition(result_path, test_manifest)
            self.assertEqual(verification["train_test_chunk_overlap"], 0)
            self.assertEqual(verification["validation_test_chunk_overlap"], 0)

            test_manifest.write_text(
                json.dumps(
                    {
                        "clip_id": "clip-test",
                        "t0_us": 90,
                        "chunk_id": "chunk-val",
                        "artifact_path": str(root / "unused.npz"),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "validation data share source chunks"
            ):
                verify_evaluation_partition(result_path, test_manifest)

    def test_selection_rejects_repeated_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_checkpoint(root / "first", seed=2026, best_loss=0.4)
            write_checkpoint(root / "second", seed=2026, best_loss=0.3)
            with self.assertRaisesRegex(ValueError, "distinct random seeds"):
                select_checkpoint(
                    (("first", root / "first"), ("second", root / "second"))
                )


if __name__ == "__main__":
    unittest.main()
