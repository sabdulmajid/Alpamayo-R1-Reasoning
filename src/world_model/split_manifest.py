"""Create deterministic, chunk-disjoint train/validation/test manifests."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

from .data import load_manifest, write_jsonl
from .runtime import write_json


SPLIT_NAMES = ("train", "val", "test")


def parse_ratios(value: str | Sequence[float]) -> tuple[float, float, float]:
    ratios = (
        tuple(float(part) for part in value.split(","))
        if isinstance(value, str)
        else tuple(value)
    )
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be three non-negative values")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("At least one split ratio must be positive")
    return tuple(ratio / total for ratio in ratios)  # type: ignore[return-value]


def assign_group(group_id: str, seed: int, ratios: Sequence[float]) -> str:
    """Assign a group by stable hash; independent of manifest row order."""

    normalized = parse_ratios(ratios)
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < normalized[0]:
        return "train"
    if value < normalized[0] + normalized[1]:
        return "val"
    return "test"


def split_rows(
    rows: list[dict[str, object]], seed: int, ratios: Sequence[float]
) -> dict[str, list[dict[str, object]]]:
    splits: dict[str, list[dict[str, object]]] = {name: [] for name in SPLIT_NAMES}
    group_assignments: dict[str, str] = {}
    for row in sorted(
        rows,
        key=lambda item: (
            str(item["chunk_id"]),
            str(item["clip_id"]),
            int(item["t0_us"]),
        ),
    ):
        group = str(row["chunk_id"])
        split = group_assignments.setdefault(group, assign_group(group, seed, ratios))
        splits[split].append(row)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ratios", default="0.8,0.1,0.1", help="train,val,test")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    ratios = parse_ratios(args.ratios)
    splits = split_rows(rows, args.seed, ratios)
    chunk_sets = {
        name: {str(row["chunk_id"]) for row in split} for name, split in splits.items()
    }
    if any(
        chunk_sets[left] & chunk_sets[right]
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    ):
        raise AssertionError("Internal error: chunk leakage across splits")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"seed": args.seed, "ratios": dict(zip(SPLIT_NAMES, ratios))}
    for name, split in splits.items():
        write_jsonl(split, output_dir / f"{name}.jsonl")
        summary[name] = {
            "clips": len(split),
            "chunks": len(chunk_sets[name]),
        }
    write_json(summary, output_dir / "split_summary.json")
    print((output_dir / "split_summary.json").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
