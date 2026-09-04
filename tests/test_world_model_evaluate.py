import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.world_model.evaluate import main, validate_learned_test_binding
from src.world_model.runtime import canonical_fingerprint, sha256_file


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class BindingFixture:
    def __init__(self, root: Path) -> None:
        self.checkpoint_path = (root / "best.pt").resolve()
        self.checkpoint_path.write_bytes(b"selected checkpoint bytes")
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self.manifest_path = (root / "test.jsonl").resolve()
        self.manifest_path.write_text(
            '{"clip_id":"test","t0_us":1,"chunk_id":"test-a","artifact_path":"x"}\n',
            encoding="utf-8",
        )
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.evaluation_provenance = {
            "artifact_count": 2,
            "chunk_ids": ["test-a", "test-b"],
            "dataset_fingerprint": "f" * 64,
        }
        training_configuration = {
            "epochs": 5,
            "batch_size": 4,
            "workers": 2,
            "base_channels": 16,
            "automatic_mixed_precision": True,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        }
        self.protocol_path = (root / "protocol.json").resolve()
        self.protocol = {
            "schema_version": 1,
            "status": "frozen_before_test_evaluation",
            "source_data": {},
            "split": {
                "unit": "chunk_id",
                "manifests": {
                    "test": {
                        "path": str(self.manifest_path),
                        "sha256": self.manifest_sha256,
                        "clips": 2,
                        "chunks": 2,
                        "dataset_fingerprint": self.evaluation_provenance[
                            "dataset_fingerprint"
                        ],
                    }
                },
            },
            "training": {
                "seeds": [2026, 2027],
                **training_configuration,
                "checkpoint_selection": "minimum_best_validation_loss",
                "test_metrics_used_for_selection": False,
            },
            "forecast_evaluation": {},
            "trajectory_selection": {},
            "uncertainty": {},
            "acceptance_gates": {},
        }
        self.protocol["protocol_fingerprint"] = canonical_fingerprint(self.protocol)
        _write_json(self.protocol_path, self.protocol)
        protocol_provenance = {
            "path": str(self.protocol_path),
            "sha256": sha256_file(self.protocol_path),
            "protocol_fingerprint": self.protocol["protocol_fingerprint"],
        }
        comparison_signature = {
            "protocol": protocol_provenance,
            "train_manifest_sha256": "a" * 64,
            "val_manifest_sha256": "b" * 64,
            "batch_size": 4,
            "base_channels": 16,
            "amp": True,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        }
        self.checkpoint = {
            "epoch": 4,
            "seed": 2027,
            "best_val_loss": 0.25,
            "model_config": {"base_channels": 16},
            "data_schema": {"coordinate_frame": "ego_at_t0"},
            "resume_signature": {
                **comparison_signature,
                "seed": 2027,
            },
            "run_config": {
                "epochs": 5,
                "batch_size": 4,
                "workers": 2,
                "base_channels": 16,
                "amp": True,
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
            },
            "data_provenance": {
                "train": {
                    "dataset_fingerprint": "c" * 64,
                    "chunk_ids": ["train"],
                },
                "val": {
                    "dataset_fingerprint": "d" * 64,
                    "chunk_ids": ["validation"],
                },
            },
        }
        selected_run = {
            "label": "seed-2027",
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "epoch": 4,
            "seed": 2027,
            "best_validation_loss": 0.25,
            "train_manifest_sha256": "a" * 64,
            "validation_manifest_sha256": "b" * 64,
            "train_dataset_fingerprint": "c" * 64,
            "validation_dataset_fingerprint": "d" * 64,
            "model_config": {"base_channels": 16},
            "data_schema": {"coordinate_frame": "ego_at_t0"},
            "training_configuration": training_configuration,
            "protocol": protocol_provenance,
            "comparison_signature": comparison_signature,
        }
        other_checkpoint = (root / "other.pt").resolve()
        other_checkpoint.write_bytes(b"other checkpoint bytes")
        other_run = {
            **selected_run,
            "label": "seed-2026",
            "checkpoint": str(other_checkpoint),
            "checkpoint_sha256": sha256_file(other_checkpoint),
            "seed": 2026,
            "best_validation_loss": 0.5,
        }

        def add_completion(run: dict) -> None:
            label = str(run["label"])
            latest_path = (root / f"{label}-latest.pt").resolve()
            latest_path.write_bytes(f"latest {label}".encode())
            completion = {
                "schema_version": 1,
                "status": "completed",
                "seed": run["seed"],
                "epochs": 5,
                "final_epoch": 4,
                "protocol": protocol_provenance,
                "latest_checkpoint": str(latest_path),
                "latest_checkpoint_sha256": sha256_file(latest_path),
                "best_checkpoint": run["checkpoint"],
                "best_checkpoint_sha256": run["checkpoint_sha256"],
            }
            completion["completion_fingerprint"] = canonical_fingerprint(completion)
            completion_path = (root / f"{label}-completion.json").resolve()
            _write_json(completion_path, completion)
            run["training_completion"] = {
                "path": str(completion_path),
                "sha256": sha256_file(completion_path),
                "completion_fingerprint": completion["completion_fingerprint"],
                "latest_checkpoint": str(latest_path),
                "latest_checkpoint_sha256": sha256_file(latest_path),
            }

        add_completion(selected_run)
        add_completion(other_run)
        self.selection = {
            "schema_version": 2,
            "selection_policy": "minimum_best_validation_loss",
            "test_metrics_used": False,
            "protocol": protocol_provenance,
            "selected_label": "seed-2027",
            "selected_seed": 2027,
            "selected_checkpoint": str(self.checkpoint_path),
            "selected_checkpoint_sha256": self.checkpoint_sha256,
            "selected_epoch": 4,
            "selected_validation_loss": 0.25,
            "runs": [selected_run, other_run],
        }
        self.selection["selection_fingerprint"] = canonical_fingerprint(self.selection)
        self.selection_path = (root / "selection.json").resolve()
        _write_json(self.selection_path, self.selection)
        self.audit = {
            "selection": str(self.selection_path),
            "selection_sha256": sha256_file(self.selection_path),
            "selection_fingerprint": self.selection["selection_fingerprint"],
            "protocol": protocol_provenance,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "test_manifest": str(self.manifest_path),
            "test_manifest_sha256": self.manifest_sha256,
            "test_clips": 2,
            "test_chunks": 2,
            "train_test_chunk_overlap": 0,
            "validation_test_chunk_overlap": 0,
        }
        self.audit_path = (root / "audit.json").resolve()
        _write_json(self.audit_path, self.audit)

    def rewrite_selection(self) -> None:
        payload = dict(self.selection)
        payload.pop("selection_fingerprint", None)
        self.selection["selection_fingerprint"] = canonical_fingerprint(payload)
        _write_json(self.selection_path, self.selection)
        self.audit["selection_sha256"] = sha256_file(self.selection_path)
        self.audit["selection_fingerprint"] = self.selection[
            "selection_fingerprint"
        ]
        _write_json(self.audit_path, self.audit)

    def rewrite_completion(self, run_index: int) -> None:
        run = self.selection["runs"][run_index]
        record = run["training_completion"]
        path = Path(record["path"])
        completion = json.loads(path.read_text(encoding="utf-8"))
        completion["seed"] = run["seed"]
        completion.pop("completion_fingerprint", None)
        completion["completion_fingerprint"] = canonical_fingerprint(completion)
        _write_json(path, completion)
        record["sha256"] = sha256_file(path)
        record["completion_fingerprint"] = completion["completion_fingerprint"]

    def validate(self) -> dict:
        return validate_learned_test_binding(
            protocol=self.protocol_path,
            selection_record=self.selection_path,
            evaluation_audit=self.audit_path,
            checkpoint_path=self.checkpoint_path,
            checkpoint_sha256=self.checkpoint_sha256,
            checkpoint=self.checkpoint,
            manifest_path=self.manifest_path,
            manifest_sha256=self.manifest_sha256,
            evaluation_provenance=self.evaluation_provenance,
        )


class LearnedTestBindingTest(unittest.TestCase):
    def test_accepts_exact_selection_checkpoint_and_partition_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BindingFixture(Path(temporary))
            binding = fixture.validate()
            self.assertEqual(
                binding["checkpoint_selection"]["sha256"],
                sha256_file(fixture.selection_path),
            )
            self.assertEqual(
                binding["evaluation_audit"]["sha256"],
                sha256_file(fixture.audit_path),
            )
            self.assertEqual(
                binding["checkpoint_selection"]["selection_fingerprint"],
                fixture.selection["selection_fingerprint"],
            )
            self.assertEqual(binding["protocol"]["path"], str(fixture.protocol_path))

    def test_rejects_incomplete_or_nonwinning_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BindingFixture(Path(temporary))
            fixture.selection["runs"] = fixture.selection["runs"][:1]
            fixture.rewrite_selection()
            with self.assertRaisesRegex(ValueError, "at least two"):
                fixture.validate()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = BindingFixture(Path(temporary))
            selected, competitor = fixture.selection["runs"]
            competitor["best_validation_loss"] = 0.1
            fixture.selection["runs"] = [competitor, selected]
            fixture.rewrite_selection()
            with self.assertRaisesRegex(ValueError, "minimum validation-loss"):
                fixture.validate()

    def test_rejects_duplicate_or_incomparable_runs_and_invalid_losses(self) -> None:
        mutations = (
            ("label", "labels must be unique"),
            ("seed", "distinct random seeds"),
            ("configuration", "runs differ in model_config"),
            ("loss", "invalid validation loss"),
        )
        for mutation, message in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = BindingFixture(Path(temporary))
                first, second = fixture.selection["runs"]
                if mutation == "label":
                    second["label"] = first["label"]
                elif mutation == "seed":
                    second["seed"] = first["seed"]
                    fixture.rewrite_completion(1)
                elif mutation == "configuration":
                    second["model_config"] = {"base_channels": 32}
                else:
                    second["best_validation_loss"] = "not-finite"
                fixture.rewrite_selection()
                with self.assertRaisesRegex(ValueError, message):
                    fixture.validate()

    def test_rejects_protocol_manifest_and_training_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BindingFixture(Path(temporary))
            copied_protocol = (Path(temporary) / "copied-protocol.json").resolve()
            copied_protocol.write_bytes(fixture.protocol_path.read_bytes())
            with self.assertRaisesRegex(ValueError, "requested frozen protocol"):
                validate_learned_test_binding(
                    protocol=copied_protocol,
                    selection_record=fixture.selection_path,
                    evaluation_audit=fixture.audit_path,
                    checkpoint_path=fixture.checkpoint_path,
                    checkpoint_sha256=fixture.checkpoint_sha256,
                    checkpoint=fixture.checkpoint,
                    manifest_path=fixture.manifest_path,
                    manifest_sha256=fixture.manifest_sha256,
                    evaluation_provenance=fixture.evaluation_provenance,
                )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = BindingFixture(Path(temporary))
            fixture.protocol["training"]["epochs"] = 6
            fixture.protocol.pop("protocol_fingerprint")
            fixture.protocol["protocol_fingerprint"] = canonical_fingerprint(
                fixture.protocol
            )
            _write_json(fixture.protocol_path, fixture.protocol)
            with self.assertRaisesRegex(ValueError, "requested frozen protocol"):
                fixture.validate()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = BindingFixture(Path(temporary))
            copied_manifest = (Path(temporary) / "copied-test.jsonl").resolve()
            copied_manifest.write_bytes(fixture.manifest_path.read_bytes())
            with self.assertRaisesRegex(ValueError, "test manifest path differs"):
                validate_learned_test_binding(
                    protocol=fixture.protocol_path,
                    selection_record=fixture.selection_path,
                    evaluation_audit=fixture.audit_path,
                    checkpoint_path=fixture.checkpoint_path,
                    checkpoint_sha256=fixture.checkpoint_sha256,
                    checkpoint=fixture.checkpoint,
                    manifest_path=copied_manifest,
                    manifest_sha256=sha256_file(copied_manifest),
                    evaluation_provenance=fixture.evaluation_provenance,
                )

    def test_cli_requires_protocol_only_for_learned_test(self) -> None:
        cases = (
            (
                [
                    "evaluate",
                    "--manifest",
                    "missing.jsonl",
                    "--method",
                    "learned",
                    "--checkpoint",
                    "missing.pt",
                    "--selection-record",
                    "selection.json",
                    "--evaluation-audit",
                    "audit.json",
                    "--output",
                    "output.json",
                ],
                "requires --protocol",
            ),
            (
                [
                    "evaluate",
                    "--manifest",
                    "missing.jsonl",
                    "--method",
                    "persistence",
                    "--protocol",
                    "protocol.json",
                    "--output",
                    "output.json",
                ],
                "only valid for learned test evaluation",
            ),
        )
        for arguments, message in cases:
            with self.subTest(message=message), patch.object(sys, "argv", arguments):
                with self.assertRaisesRegex(ValueError, message):
                    main()

    def test_rejects_selection_payload_without_matching_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BindingFixture(Path(temporary))
            fixture.selection["selected_epoch"] = 5
            _write_json(fixture.selection_path, fixture.selection)
            fixture.audit["selection_sha256"] = sha256_file(fixture.selection_path)
            _write_json(fixture.audit_path, fixture.audit)
            with self.assertRaisesRegex(ValueError, "fingerprint does not match"):
                fixture.validate()

    def test_requires_selection_fingerprint_in_both_records(self) -> None:
        for record in ("selection", "audit"):
            with (
                self.subTest(record=record),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fixture = BindingFixture(Path(temporary))
                if record == "selection":
                    fixture.selection.pop("selection_fingerprint")
                    _write_json(fixture.selection_path, fixture.selection)
                    fixture.audit["selection_sha256"] = sha256_file(
                        fixture.selection_path
                    )
                else:
                    fixture.audit.pop("selection_fingerprint")
                _write_json(fixture.audit_path, fixture.audit)
                with self.assertRaisesRegex(
                    ValueError, "missing 'selection_fingerprint'"
                ):
                    fixture.validate()

    def test_rejects_checkpoint_path_sha_and_metadata_changes(self) -> None:
        mutations = (
            ("path", "path .* does not match"),
            ("sha", "SHA-256 does not match"),
            ("epoch", "epoch or seed differs"),
            ("provenance", "train_dataset_fingerprint"),
        )
        for mutation, message in mutations:
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fixture = BindingFixture(Path(temporary))
                if mutation == "path":
                    fixture.selection["selected_checkpoint"] = str(
                        (Path(temporary) / "different.pt").resolve()
                    )
                elif mutation == "sha":
                    fixture.selection["selected_checkpoint_sha256"] = "0" * 64
                elif mutation == "epoch":
                    fixture.selection["selected_epoch"] = 5
                    fixture.selection["runs"][0]["epoch"] = 5
                else:
                    fixture.selection["runs"][0]["train_dataset_fingerprint"] = "0" * 64
                fixture.rewrite_selection()
                with self.assertRaisesRegex(ValueError, message):
                    fixture.validate()

    def test_rejects_audit_path_hash_count_and_overlap_changes(self) -> None:
        mutations = (
            ("selection_path", "selection record.*does not match"),
            ("selection_sha", "selection record bytes"),
            ("selection_fingerprint", "selection fingerprint does not match"),
            ("checkpoint_path", "checkpoint path.*does not match"),
            ("checkpoint_sha", "checkpoint bytes"),
            ("manifest_path", "manifest path.*does not match"),
            ("manifest_sha", "test manifest bytes"),
            ("test_count", "test counts differ"),
            ("train_overlap", "train_test_chunk_overlap=0"),
            ("validation_overlap", "validation_test_chunk_overlap=0"),
        )
        for mutation, message in mutations:
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fixture = BindingFixture(Path(temporary))
                if mutation == "selection_path":
                    fixture.audit["selection"] = str(
                        (Path(temporary) / "different-selection.json").resolve()
                    )
                elif mutation == "selection_sha":
                    fixture.audit["selection_sha256"] = "0" * 64
                elif mutation == "selection_fingerprint":
                    fixture.audit["selection_fingerprint"] = "0" * 64
                elif mutation == "checkpoint_path":
                    fixture.audit["checkpoint"] = str(
                        (Path(temporary) / "different.pt").resolve()
                    )
                elif mutation == "checkpoint_sha":
                    fixture.audit["checkpoint_sha256"] = "0" * 64
                elif mutation == "manifest_path":
                    fixture.audit["test_manifest"] = str(
                        (Path(temporary) / "different-test.jsonl").resolve()
                    )
                elif mutation == "manifest_sha":
                    fixture.audit["test_manifest_sha256"] = "0" * 64
                elif mutation == "test_count":
                    fixture.audit["test_clips"] = 1
                elif mutation == "train_overlap":
                    fixture.audit["train_test_chunk_overlap"] = 1
                else:
                    fixture.audit["validation_test_chunk_overlap"] = 1
                _write_json(fixture.audit_path, fixture.audit)
                with self.assertRaisesRegex(ValueError, message):
                    fixture.validate()


if __name__ == "__main__":
    unittest.main()
