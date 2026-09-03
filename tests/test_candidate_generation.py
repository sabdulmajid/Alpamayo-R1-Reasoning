from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.export_candidate_metrics import flatten_records
from src.generate_candidates import (
    GenerationConfig,
    artifact_key,
    atomic_write_json,
    atomic_write_npz,
    candidate_seed_schedule,
    candidate_times_s,
    concatenate_candidate_rollouts,
    completion_status,
    compute_candidate_metrics,
    derive_candidate_seed,
    derive_clip_seed,
    file_sha256,
    fresh_rollout_inputs,
    parse_args,
    select_shard,
    validate_outputs,
)


class CandidateGenerationTest(unittest.TestCase):
    @staticmethod
    def _write_artifact(
        path: Path,
        config: GenerationConfig,
        clip_id: str = "clip-a",
        t0_us: int = 5,
        include_candidate_seeds: bool = True,
    ) -> None:
        arrays = {
            "pred_xyz": np.zeros((config.num_candidates, 64, 3), dtype=np.float32),
            "pred_rot": np.zeros((config.num_candidates, 64, 3, 3), dtype=np.float32),
            "gt_xyz": np.zeros((64, 3), dtype=np.float32),
            "gt_rot": np.zeros((64, 3, 3), dtype=np.float32),
            "candidate_coc": np.full(config.num_candidates, "reasoning"),
            "candidate_meta_action": np.full(config.num_candidates, ""),
            "candidate_answer": np.full(config.num_candidates, ""),
            "candidate_times_s": candidate_times_s(),
            "config_fingerprint": np.asarray(config.fingerprint),
            "schema_version": np.asarray(1, dtype=np.int32),
            "clip_id": np.asarray(clip_id),
            "t0_us": np.asarray(t0_us, dtype=np.int64),
            "clip_seed": np.asarray(
                derive_clip_seed(config.base_seed, clip_id, t0_us), dtype=np.int64
            ),
        }
        if include_candidate_seeds:
            arrays["candidate_seeds"] = candidate_seed_schedule(config, clip_id, t0_us)
        atomic_write_npz(path, **arrays)

    @staticmethod
    def _record_payload(
        config: GenerationConfig,
        artifact_path: Path,
        clip_id: str = "clip-a",
        t0_us: int = 5,
    ) -> dict:
        key = artifact_key(clip_id, t0_us)
        seeds = candidate_seed_schedule(config, clip_id, t0_us).tolist()
        zero_metrics = asdict(
            compute_candidate_metrics(
                np.zeros((64, 3), dtype=np.float32),
                np.zeros((64, 3), dtype=np.float32),
            )
        )
        return {
            "schema_version": 1,
            "clip_id": clip_id,
            "t0_us": t0_us,
            "artifact_path": f"artifacts/{key}.npz",
            "artifact_sha256": file_sha256(artifact_path),
            "hour_bucket": "day",
            "clip_seed": derive_clip_seed(config.base_seed, clip_id, t0_us),
            "config": asdict(config),
            "config_fingerprint": config.fingerprint,
            "candidate_seeds": seeds,
            "oracle_min_ade_candidate_index": 0,
            "candidates": [
                {
                    "candidate_index": index,
                    "candidate_seed": seed,
                    "coc": "reasoning",
                    "meta_action": "",
                    "answer": "",
                    "metrics": zero_metrics,
                    "is_oracle_min_ade": index == 0,
                }
                for index, seed in enumerate(seeds)
            ],
        }

    def test_clip_seed_is_stable_and_clip_specific(self) -> None:
        seed = derive_clip_seed(42, "clip-a", 5_000_000)
        self.assertEqual(seed, derive_clip_seed(42, "clip-a", 5_000_000))
        self.assertNotEqual(seed, derive_clip_seed(42, "clip-b", 5_000_000))
        self.assertNotEqual(seed, derive_clip_seed(43, "clip-a", 5_000_000))

    def test_candidate_seed_schedule_is_stable_distinct_and_indexed(self) -> None:
        config = GenerationConfig(num_candidates=6, base_seed=42)
        seeds = candidate_seed_schedule(config, "clip-a", 5_000_000)
        np.testing.assert_array_equal(
            seeds, candidate_seed_schedule(config, "clip-a", 5_000_000)
        )
        self.assertEqual(len(np.unique(seeds)), 6)
        self.assertEqual(
            int(seeds[3]), derive_candidate_seed(42, "clip-a", 5_000_000, 3)
        )
        self.assertFalse(
            np.array_equal(seeds, candidate_seed_schedule(config, "clip-b", 5_000_000))
        )

    def test_shards_are_disjoint_and_complete(self) -> None:
        frame = pd.DataFrame({"clip_id": [f"c{i}" for i in range(11)]})
        shards = [select_shard(frame, index, 3) for index in range(3)]
        values = [set(shard["clip_id"]) for shard in shards]
        self.assertFalse(values[0] & values[1])
        self.assertFalse(values[0] & values[2])
        self.assertFalse(values[1] & values[2])
        self.assertEqual(set().union(*values), set(frame["clip_id"]))

    def test_trajectory_metrics_include_the_t0_ego_pose(self) -> None:
        pred = np.column_stack((np.arange(2, 7), np.zeros(5), np.zeros(5))).astype(
            float
        )
        metrics = compute_candidate_metrics(pred, pred, dt_s=1.0)
        self.assertEqual(metrics.ade_m, 0.0)
        self.assertEqual(metrics.fde_m, 0.0)
        self.assertAlmostEqual(metrics.mean_speed_mps, 1.2)
        self.assertAlmostEqual(metrics.mean_abs_accel_mps2, 0.25)
        self.assertAlmostEqual(metrics.mean_abs_jerk_mps3, 1.0 / 3.0)
        self.assertEqual(metrics.max_curvature_inv_m, 0.0)
        self.assertAlmostEqual(metrics.path_length_m, 6.0)

    def test_output_validation_rejects_nonfinite_values(self) -> None:
        xyz = np.zeros((2, 64, 3), dtype=np.float32)
        rot = np.zeros((2, 64, 3, 3), dtype=np.float32)
        gt_xyz = np.zeros((64, 3), dtype=np.float32)
        gt_rot = np.zeros((64, 3, 3), dtype=np.float32)
        validate_outputs(xyz, rot, gt_xyz, gt_rot, ["a", "b"], [11, 22], 2)
        xyz[1, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_outputs(xyz, rot, gt_xyz, gt_rot, ["a", "b"], [11, 22], 2)

    def test_sequential_rollouts_preserve_candidate_order(self) -> None:
        rollouts = []
        for index, seed in enumerate((101, 202, 303)):
            rollouts.append(
                {
                    "pred_xyz": np.full((1, 64, 3), index, dtype=np.float32),
                    "pred_rot": np.full((1, 64, 3, 3), index, dtype=np.float32),
                    "candidate_coc": [f"reasoning-{index}"],
                    "candidate_meta_action": [f"action-{index}"],
                    "candidate_answer": [f"answer-{index}"],
                    "candidate_seeds": np.asarray([seed], dtype=np.int64),
                }
            )
        combined = concatenate_candidate_rollouts(rollouts)
        self.assertEqual(combined["pred_xyz"].shape, (3, 64, 3))
        self.assertEqual(combined["pred_xyz"][:, 0, 0].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(
            combined["candidate_coc"],
            ["reasoning-0", "reasoning-1", "reasoning-2"],
        )
        self.assertEqual(combined["candidate_seeds"].tolist(), [101, 202, 303])

    def test_each_rollout_receives_fresh_input_ids_mapping(self) -> None:
        template = {
            "tokenized_data": {
                "input_ids": np.asarray([[1, 2, 3]]),
                "attention_mask": np.asarray([[1, 1, 1]]),
            },
            "ego_history_xyz": np.zeros((1, 1, 1, 3)),
        }
        received = []

        def mutating_rollout(inputs: dict) -> None:
            received.append(inputs["tokenized_data"].pop("input_ids").copy())

        for _ in range(6):
            mutating_rollout(fresh_rollout_inputs(template))

        self.assertEqual(len(received), 6)
        self.assertIn("input_ids", template["tokenized_data"])
        for input_ids in received:
            np.testing.assert_array_equal(input_ids, [[1, 2, 3]])

    def test_atomic_artifact_can_be_resumed(self) -> None:
        config = GenerationConfig(num_candidates=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = artifact_key("clip-a", 5)
            artifact = root / "artifacts" / f"{key}.npz"
            record = root / "records" / f"{key}.json"
            self._write_artifact(artifact, config)
            atomic_write_json(record, self._record_payload(config, artifact))
            self.assertEqual(
                completion_status(record, artifact, config, "clip-a", 5),
                (True, None),
            )
            changed = GenerationConfig(num_candidates=3)
            complete, reason = completion_status(record, artifact, changed, "clip-a", 5)
            self.assertFalse(complete)
            self.assertIn("different generation configuration", reason or "")

            complete, reason = completion_status(record, artifact, config, "clip-b", 5)
            self.assertFalse(complete)
            self.assertIn("record identity", reason or "")

            self._write_artifact(artifact, config, clip_id="clip-b")
            atomic_write_json(record, self._record_payload(config, artifact))
            complete, reason = completion_status(record, artifact, config, "clip-a", 5)
            self.assertFalse(complete)
            self.assertIn("artifact identity", reason or "")

    def test_resume_requires_candidate_seed_schema(self) -> None:
        config = GenerationConfig(num_candidates=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = artifact_key("clip-a", 5)
            artifact = root / "artifacts" / f"{key}.npz"
            record = root / "records" / f"{key}.json"
            self._write_artifact(artifact, config)
            incomplete_record = self._record_payload(config, artifact)
            incomplete_record.pop("candidate_seeds")
            atomic_write_json(record, incomplete_record)
            complete, reason = completion_status(record, artifact, config, "clip-a", 5)
            self.assertFalse(complete)
            self.assertIn("candidate seeds", reason or "")

            self._write_artifact(artifact, config, include_candidate_seeds=False)
            atomic_write_json(record, self._record_payload(config, artifact))
            complete, reason = completion_status(record, artifact, config, "clip-a", 5)
            self.assertFalse(complete)
            self.assertIn("candidate_seeds", reason or "")

            self._write_artifact(artifact, config, t0_us=6)
            atomic_write_json(record, self._record_payload(config, artifact))
            complete, reason = completion_status(record, artifact, config, "clip-a", 5)
            self.assertFalse(complete)
            self.assertIn("artifact identity", reason or "")

    def test_resume_rejects_wrong_artifact_reference(self) -> None:
        config = GenerationConfig(num_candidates=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = artifact_key("clip-a", 5)
            artifact = root / "artifacts" / f"{key}.npz"
            record = root / "records" / f"{key}.json"
            self._write_artifact(artifact, config)
            payload = self._record_payload(config, artifact)
            payload["artifact_path"] = "artifacts/not-this-clip.npz"
            atomic_write_json(record, payload)
            complete, reason = completion_status(record, artifact, config, "clip-a", 5)
            self.assertFalse(complete)
            self.assertIn("artifact_path", reason or "")

    def test_resume_rejects_changed_artifact_contents(self) -> None:
        config = GenerationConfig(num_candidates=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = artifact_key("clip-a", 5)
            artifact = root / "artifacts" / f"{key}.npz"
            record = root / "records" / f"{key}.json"
            self._write_artifact(artifact, config)
            atomic_write_json(record, self._record_payload(config, artifact))

            with np.load(artifact, allow_pickle=False) as saved:
                arrays = {name: saved[name] for name in saved.files}
            arrays["pred_xyz"] = arrays["pred_xyz"].copy()
            arrays["pred_xyz"][0, :, 0] += 100.0
            atomic_write_npz(artifact, **arrays)

            complete, reason = completion_status(
                record, artifact, config, "clip-a", 5
            )
            self.assertFalse(complete)
            self.assertIn("artifact_sha256", reason or "")

    def test_resume_rejects_record_text_or_metrics_not_in_artifact(self) -> None:
        config = GenerationConfig(num_candidates=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = artifact_key("clip-a", 5)
            artifact = root / "artifacts" / f"{key}.npz"
            record = root / "records" / f"{key}.json"
            self._write_artifact(artifact, config)

            payload = self._record_payload(config, artifact)
            payload["candidates"][0]["coc"] = "changed"
            atomic_write_json(record, payload)
            complete, reason = completion_status(
                record, artifact, config, "clip-a", 5
            )
            self.assertFalse(complete)
            self.assertIn("coc", reason or "")

            payload = self._record_payload(config, artifact)
            payload["candidates"][0]["metrics"]["ade_m"] = 1.0
            atomic_write_json(record, payload)
            complete, reason = completion_status(
                record, artifact, config, "clip-a", 5
            )
            self.assertFalse(complete)
            self.assertIn("metrics", reason or "")

    def test_candidate_times_cover_model_horizon(self) -> None:
        times = candidate_times_s()
        self.assertEqual(times.shape, (64,))
        self.assertAlmostEqual(float(times[0]), 0.1, places=6)
        self.assertAlmostEqual(float(times[-1]), 6.4, places=6)
        np.testing.assert_allclose(np.diff(times), 0.1, rtol=0.0, atol=1e-6)

    def test_metric_export_preserves_oracle_terminology(self) -> None:
        metrics = asdict(
            compute_candidate_metrics(
                np.zeros((4, 3), dtype=float), np.zeros((4, 3), dtype=float)
            )
        )
        record = {
            "schema_version": 1,
            "clip_id": "clip-a",
            "t0_us": 5,
            "hour_bucket": "day",
            "clip_seed": 7,
            "candidate_seeds": [11],
            "oracle_min_ade_candidate_index": 0,
            "config_fingerprint": "abc",
            "artifact_path": "artifacts/a.npz",
            "artifact_sha256": "abc123",
            "candidates": [
                {
                    "candidate_index": 0,
                    "candidate_seed": 11,
                    "coc": "maintain speed",
                    "meta_action": "",
                    "answer": "",
                    "metrics": metrics,
                    "is_oracle_min_ade": True,
                }
            ],
        }
        rows = flatten_records([json.loads(json.dumps(record))])
        self.assertEqual(rows[0]["oracle_min_ade_candidate_index"], 0)
        self.assertTrue(rows[0]["is_oracle_min_ade"])
        self.assertEqual(rows[0]["candidate_seed"], 11)
        self.assertEqual(rows[0]["hour_bucket"], "day")
        self.assertEqual(rows[0]["artifact_sha256"], "abc123")

    def test_artifact_key_does_not_embed_clip_id(self) -> None:
        key = artifact_key("../unsafe/clip", 123)
        self.assertNotIn("/", key)
        self.assertNotIn("..", key)

    def test_official_candidate_defaults_are_preserved(self) -> None:
        args = parse_args([])
        self.assertEqual(args.num_candidates, 6)
        self.assertEqual(args.max_generation_length, 256)
        self.assertEqual(args.temperature, 0.6)
        self.assertEqual(GenerationConfig().samples_per_rollout, 1)
        with self.assertRaisesRegex(ValueError, "samples_per_rollout=1"):
            GenerationConfig(samples_per_rollout=2).validate()


if __name__ == "__main__":
    unittest.main()
