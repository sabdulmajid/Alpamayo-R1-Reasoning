import json
import tempfile
import unittest
from pathlib import Path

from src.world_model.evaluate import validate_learned_test_binding
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
        self.checkpoint = {
            "epoch": 4,
            "seed": 2027,
            "best_val_loss": 0.25,
            "model_config": {"base_channels": 16},
            "data_schema": {"coordinate_frame": "ego_at_t0"},
            "resume_signature": {
                "train_manifest_sha256": "a" * 64,
                "val_manifest_sha256": "b" * 64,
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
        }
        other_run = {
            **selected_run,
            "label": "seed-2026",
            "checkpoint": str((root / "other.pt").resolve()),
            "checkpoint_sha256": "e" * 64,
            "seed": 2026,
            "best_validation_loss": 0.5,
        }
        self.selection = {
            "schema_version": 1,
            "selection_policy": "minimum_best_validation_loss",
            "test_metrics_used": False,
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
        self.evaluation_provenance = {
            "artifact_count": 2,
            "chunk_ids": ["test-a", "test-b"],
            "dataset_fingerprint": "f" * 64,
        }

    def rewrite_selection(self) -> None:
        payload = dict(self.selection)
        payload.pop("selection_fingerprint", None)
        self.selection["selection_fingerprint"] = canonical_fingerprint(payload)
        _write_json(self.selection_path, self.selection)
        self.audit["selection_sha256"] = sha256_file(self.selection_path)
        _write_json(self.audit_path, self.audit)

    def validate(self) -> dict:
        return validate_learned_test_binding(
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
            ("provenance", "does not match checkpoint provenance"),
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
