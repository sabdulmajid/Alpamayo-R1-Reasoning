"""Export atomic candidate records to one analysis-friendly CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


FIELDS = [
    "clip_id",
    "t0_us",
    "hour_bucket",
    "clip_seed",
    "candidate_index",
    "candidate_seed",
    "num_candidates",
    "oracle_min_ade_candidate_index",
    "is_oracle_min_ade",
    "ade_m",
    "fde_m",
    "mean_speed_mps",
    "mean_abs_accel_mps2",
    "mean_abs_jerk_mps3",
    "max_curvature_inv_m",
    "path_length_m",
    "coc",
    "meta_action",
    "answer",
    "config_fingerprint",
    "artifact_path",
    "artifact_sha256",
]


def load_records(records_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(records_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema_version") != 1:
            raise ValueError(f"unsupported schema in {path}")
        records.append(record)
    return records


def flatten_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[str, int, int]] = set()
    for record in records:
        candidates = record["candidates"]
        for candidate in candidates:
            key = (
                record["clip_id"],
                int(record["t0_us"]),
                int(candidate["candidate_index"]),
            )
            if key in seen:
                raise ValueError(f"duplicate candidate record: {key}")
            seen.add(key)
            rows.append(
                {
                    "clip_id": record["clip_id"],
                    "t0_us": record["t0_us"],
                    "hour_bucket": record["hour_bucket"],
                    "clip_seed": record["clip_seed"],
                    "candidate_index": candidate["candidate_index"],
                    "candidate_seed": candidate["candidate_seed"],
                    "num_candidates": len(candidates),
                    "oracle_min_ade_candidate_index": record[
                        "oracle_min_ade_candidate_index"
                    ],
                    "is_oracle_min_ade": candidate["is_oracle_min_ade"],
                    **candidate["metrics"],
                    "coc": candidate["coc"],
                    "meta_action": candidate["meta_action"],
                    "answer": candidate["answer"],
                    "config_fingerprint": record["config_fingerprint"],
                    "artifact_path": record["artifact_path"],
                    "artifact_sha256": record["artifact_sha256"],
                }
            )
    return sorted(
        rows, key=lambda row: (row["clip_id"], row["t0_us"], row["candidate_index"])
    )


def atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rows = flatten_records(load_records(args.records_dir))
        atomic_write_csv(args.output, rows)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {len(rows)} candidate rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
