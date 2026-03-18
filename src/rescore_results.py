"""
Re-score saved experiment results with updated text parser patterns.

Reads the original CSV (cot_text + trajectory execution classifications),
re-parses texts with expanded patterns, re-computes compatibility scores,
and saves a new CSV + comparison summary. No GPU needed.
"""

import csv
import sys
import os
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from mismatch_scorer import (
    TextReasoningParser,
    LongitudinalIntent, LateralIntent,
    LongitudinalExecution, LateralExecution,
    LONGITUDINAL_COMPAT, LATERAL_COMPAT,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = str(PROJECT_ROOT / "results/mismatch_8563/mismatch_results_20260226_062813.csv")
OUTPUT_CSV = str(PROJECT_ROOT / "results/mismatch_8563/rescored_v3.csv")


def classify_mismatch(score: float, long_spec: bool, lat_spec: bool) -> str:
    if not long_spec and not lat_spec:
        return "unclassified_intent"
    if score < 0.3:
        return "consistent"
    if score < 0.6:
        return "partial_mismatch"
    return "severe_mismatch"


def main():
    parser = TextReasoningParser()
    rows_in = []

    with open(INPUT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_in.append(row)

    print(f"Loaded {len(rows_in)} rows from {INPUT_CSV}")
    print()

    # --- Old vs new comparison ---
    old_types = Counter()
    new_types = Counter()
    old_long_unspec = 0
    new_long_unspec = 0
    old_lat_unspec = 0
    new_lat_unspec = 0
    reclassified = []

    rescored_rows = []

    for row in rows_in:
        cot = row["cot_text"]
        old_li = row["long_intent"]
        old_la = row["lat_intent"]
        old_le = row["long_execution"]
        old_lae = row["lat_execution"]
        old_type = row["mismatch_type"]
        old_types[old_type] += 1

        if old_li == "unspecified":
            old_long_unspec += 1
        if old_la == "unspecified":
            old_lat_unspec += 1

        new_long, new_lat, new_conf = parser.parse(cot)
        long_spec = new_long != LongitudinalIntent.UNSPECIFIED
        lat_spec = new_lat != LateralIntent.UNSPECIFIED

        if new_long == LongitudinalIntent.UNSPECIFIED:
            new_long_unspec += 1
        if new_lat == LateralIntent.UNSPECIFIED:
            new_lat_unspec += 1

        long_exec = LongitudinalExecution(old_le)
        lat_exec = LateralExecution(old_lae)

        long_compat = (
            LONGITUDINAL_COMPAT.get(new_long, {}).get(long_exec, 0.5)
            if long_spec else None
        )
        lat_compat = (
            LATERAL_COMPAT.get(new_lat, {}).get(lat_exec, 0.5)
            if lat_spec else None
        )

        if long_spec and lat_spec:
            combined = 0.5 * long_compat + 0.5 * lat_compat
        elif long_spec:
            combined = long_compat
        elif lat_spec:
            combined = lat_compat
        else:
            combined = 0.5

        mismatch = round(1.0 - combined, 3)
        mtype = classify_mismatch(mismatch, long_spec, lat_spec)
        new_types[mtype] += 1

        changed = (old_type != mtype) or (old_li != new_long.value) or (old_la != new_lat.value)
        if changed:
            reclassified.append({
                "cot": cot[:80],
                "old_long": old_li, "new_long": new_long.value,
                "old_lat": old_la, "new_lat": new_lat.value,
                "old_type": old_type, "new_type": mtype,
                "old_score": row["mismatch_score"], "new_score": mismatch,
            })

        out = dict(row)
        out["long_intent"] = new_long.value
        out["lat_intent"] = new_lat.value
        out["long_match"] = long_compat if long_compat is not None else 0.5
        out["lat_match"] = lat_compat if lat_compat is not None else 0.5
        out["mismatch_score"] = mismatch
        out["mismatch_type"] = mtype
        out["confidence"] = new_conf
        rescored_rows.append(out)

    # --- Write rescored CSV ---
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rescored_rows[0].keys())
        writer.writeheader()
        writer.writerows(rescored_rows)
    print(f"Rescored results saved to {OUTPUT_CSV}")
    print()

    # --- Summary ---
    n = len(rows_in)
    print("=" * 60)
    print("COMPARISON: old patterns (v2) → new patterns (v3)")
    print("=" * 60)
    print()

    print("Mismatch type distribution:")
    all_types = sorted(set(list(old_types.keys()) + list(new_types.keys())))
    for t in all_types:
        o = old_types.get(t, 0)
        nw = new_types.get(t, 0)
        delta = nw - o
        arrow = "→"
        print(f"  {t:25s}  {o:3d} ({o/n*100:4.1f}%) {arrow} {nw:3d} ({nw/n*100:4.1f}%)  [{delta:+d}]")
    print()

    print("Unspecified intent counts:")
    print(f"  Longitudinal unspecified:  {old_long_unspec:3d} → {new_long_unspec:3d}  [{new_long_unspec - old_long_unspec:+d}]")
    print(f"  Lateral unspecified:       {old_lat_unspec:3d} → {new_lat_unspec:3d}  [{new_lat_unspec - old_lat_unspec:+d}]")
    print()

    # New intent distributions
    long_intents = Counter(r["long_intent"] for r in rescored_rows)
    lat_intents = Counter(r["lat_intent"] for r in rescored_rows)
    print("New longitudinal intent distribution:")
    for k, v in long_intents.most_common():
        print(f"  {k:15s}  {v:3d}")
    print()
    print("New lateral intent distribution:")
    for k, v in lat_intents.most_common():
        print(f"  {k:15s}  {v:3d}")
    print()

    # Mean scores
    valid = [r for r in rescored_rows if r["mismatch_type"] != "unclassified_intent"]
    if valid:
        mean_mm = sum(float(r["mismatch_score"]) for r in valid) / len(valid)
        mean_lm = sum(float(r["long_match"]) for r in valid) / len(valid)
        mean_la = sum(float(r["lat_match"]) for r in valid) / len(valid)
        mean_ade = sum(float(r["ade_meters"]) for r in valid) / len(valid)
        print(f"Classifiable clips: {len(valid)}/{n}")
        print(f"  Mean mismatch:       {mean_mm:.3f}")
        print(f"  Mean long match:     {mean_lm:.3f}")
        print(f"  Mean lat match:      {mean_la:.3f}")
        print(f"  Mean ADE:            {mean_ade:.2f} m")
        print()

    # By scenario type
    scenario_groups = {}
    for r in rescored_rows:
        st = r["scenario_type"]
        scenario_groups.setdefault(st, []).append(r)

    print("By time-of-day:")
    for st in ["night", "morning", "midday", "afternoon", "evening"]:
        group = scenario_groups.get(st, [])
        if not group:
            continue
        mean_mm = sum(float(r["mismatch_score"]) for r in group) / len(group)
        mean_ade = sum(float(r["ade_meters"]) for r in group) / len(group)
        n_uncl = sum(1 for r in group if r["mismatch_type"] == "unclassified_intent")
        print(f"  {st:12s}  mismatch={mean_mm:.3f}  ADE={mean_ade:.2f}m  n={len(group)}  unclassified={n_uncl}")
    print()

    # Show reclassified examples
    print(f"Reclassified clips: {len(reclassified)}/{n}")
    for i, r in enumerate(reclassified[:20]):
        print(f"  {i+1:2d}. \"{r['cot'][:65]}...\"")
        print(f"      long: {r['old_long']:12s} → {r['new_long']:12s}")
        print(f"      lat:  {r['old_lat']:12s} → {r['new_lat']:12s}")
        print(f"      type: {r['old_type']:20s} → {r['new_type']:20s}  score: {r['old_score']} → {r['new_score']}")


if __name__ == "__main__":
    main()
