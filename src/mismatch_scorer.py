"""
Mismatch Scorer v2: Dual-axis decomposition of reasoning vs. action.

Instead of forcing both text intent and trajectory into a single action
category, we decompose each into two independent axes:

  - Longitudinal (speed/acceleration behavior)
  - Lateral (steering/lane position behavior)

This prevents false "unknown" classifications when the VLA model uses
nuanced driving language like "nudge to the left to increase clearance"
— which is purely a lateral intent with no explicit speed instruction.

Each axis is scored independently, and the final mismatch score is a
weighted combination of whatever axes the CoC text actually specifies.
"""

import re
import numpy as np
from dataclasses import dataclass
from enum import Enum


# ── Longitudinal action space ────────────────────────────────────────

class LongitudinalIntent(Enum):
    STOP = "stop"
    SLOW_DOWN = "slow_down"
    MAINTAIN = "maintain"
    ACCELERATE = "accelerate"
    UNSPECIFIED = "unspecified"


class LongitudinalExecution(Enum):
    STOPPED = "stopped"
    DECELERATING = "decelerating"
    CONSTANT_SPEED = "constant_speed"
    ACCELERATING = "accelerating"


# ── Lateral action space ─────────────────────────────────────────────

class LateralIntent(Enum):
    NUDGE_LEFT = "nudge_left"
    NUDGE_RIGHT = "nudge_right"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    HOLD_LANE = "hold_lane"
    UNSPECIFIED = "unspecified"


class LateralExecution(Enum):
    SHIFT_LEFT = "shift_left"
    SHIFT_RIGHT = "shift_right"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    STRAIGHT = "straight"


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class TrajectoryFeatures:
    initial_speed: float        # m/s
    final_speed: float          # m/s
    mean_speed: float           # m/s
    speed_change: float         # final - initial, m/s
    mean_acceleration: float    # m/s^2
    total_lateral_shift: float  # meters, positive = left in ego frame
    max_lateral_deviation: float
    total_distance: float       # meters
    longitudinal: LongitudinalExecution
    lateral: LateralExecution


@dataclass
class MismatchResult:
    longitudinal_intent: LongitudinalIntent
    lateral_intent: LateralIntent
    text_confidence: float
    longitudinal_execution: LongitudinalExecution
    lateral_execution: LateralExecution
    trajectory_features: TrajectoryFeatures
    longitudinal_match: float   # 0..1 compatibility on speed axis
    lateral_match: float        # 0..1 compatibility on steering axis
    mismatch_score: float       # 0 = perfect match, 1 = total mismatch
    mismatch_type: str


# ── Text pattern tables ──────────────────────────────────────────────

LONGITUDINAL_PATTERNS = {
    LongitudinalIntent.STOP: [
        r"\bstop\b", r"\bstopping\b", r"\bcome to a stop\b",
        r"\bhalt\b", r"\bfull stop\b",
        r"\bbring .{0,20}to .{0,10}stop\b",
        r"\bemergency (stop|brake)\b", r"\bbrake hard\b",
    ],
    LongitudinalIntent.SLOW_DOWN: [
        r"\bslow(ing)? down\b", r"\bdecelerat\w*\b", r"\breduce speed\b",
        r"\bbrake\b", r"\bbraking\b", r"\bease off\b",
        r"\bcreep\b", r"\bcoast\b",
        r"\bcautious(ly)? approach\b", r"\bslow(er)? approach\b",
        r"\breduce .{0,10}pace\b",
        r"\blet off .{0,5}(gas|throttle)\b",
        r"\badapt speed\b",
        r"\badjust speed\b",
        r"\byield\w*\b",
    ],
    LongitudinalIntent.MAINTAIN: [
        r"\bmaintain .{0,10}(speed|pace|velocity)\b",
        r"\bkeep .{0,10}(speed|pace|velocity)\b",
        r"\bcruise\b", r"\bhold .{0,10}speed\b",
        r"\bsteady (speed|pace)\b", r"\bconstant speed\b",
        r"\bproceed\b", r"\bcontinue\b",
        r"\bkeep .{0,15}(distance|gap)\b",
        r"\bmaintain .{0,15}(distance|gap|following)\b",
        r"\bsafe (distance|gap|following)\b",
    ],
    LongitudinalIntent.ACCELERATE: [
        r"\baccelerat\w*\b", r"\bspeed up\b",
        r"\bincrease speed\b", r"\bgo faster\b",
        r"\bpick up speed\b", r"\bgain speed\b",
        r"\bresume\b",
    ],
}

LATERAL_PATTERNS = {
    LateralIntent.NUDGE_LEFT: [
        r"\bnudge .{0,10}(to the )?left\b",
        r"\bshift .{0,10}(to the )?left\b",
        r"\bsteer .{0,5}left\b",
        r"\bveer .{0,5}left\b",
        r"\bmove .{0,10}(slightly )?(to the )?left\b",
        r"\badjust .{0,15}(to the )?left\b",
        r"\bswerve .{0,5}left\b",
        r"\boffset .{0,10}(to the )?left\b",
        r"\bleft.{0,3}hand curve\b",
        r"\bleft curve\b",
        r"\bcurve .{0,10}(to the )?left\b",
    ],
    LateralIntent.NUDGE_RIGHT: [
        r"\bnudge .{0,10}(to the )?right\b",
        r"\bshift .{0,10}(to the )?right\b",
        r"\bsteer .{0,5}right\b",
        r"\bveer .{0,5}right\b",
        r"\bmove .{0,10}(slightly )?(to the )?right\b",
        r"\badjust .{0,15}(to the )?right\b",
        r"\bswerve .{0,5}right\b",
        r"\boffset .{0,10}(to the )?right\b",
        r"\bright.{0,3}hand curve\b",
        r"\bright curve\b",
        r"\bcurve .{0,10}(to the )?right\b",
        r"\bright exit\b",
    ],
    LateralIntent.LANE_CHANGE_LEFT: [
        r"\b(change|switch) .{0,15}(to .{0,10})?left .{0,5}lane\b",
        r"\bmerge .{0,15}left\b",
        r"\bleft lane change\b",
        r"\bmove .{0,15}(to .{0,10})?left lane\b",
        r"\blane change .{0,10}(to the )?left\b",
        r"\bturn left\b",
    ],
    LateralIntent.LANE_CHANGE_RIGHT: [
        r"\b(change|switch) .{0,15}(to .{0,10})?right .{0,5}lane\b",
        r"\bmerge .{0,15}right\b",
        r"\bright lane change\b",
        r"\bmove .{0,15}(to .{0,10})?right lane\b",
        r"\blane change .{0,10}(to the )?right\b",
        r"\bturn right\b",
        r"\bsplit .{0,10}(to the )?right\b",
    ],
    LateralIntent.HOLD_LANE: [
        r"\bstay in .{0,15}lane\b",
        r"\bhold .{0,10}lane\b",
        r"\bkeep .{0,15}lane\b",
        r"\bmaintain .{0,15}lane\b",
        r"\bcenter .{0,15}lane\b",
        r"\bstay .{0,10}course\b",
        r"\bin the (same )?lane\b",
        r"\bthrough the intersection\b",
        r"\bstraight traffic light\b",
        r"\bfollow the (road|route)\b",
    ],
}


# ── Compatibility matrices ───────────────────────────────────────────

LONGITUDINAL_COMPAT = {
    LongitudinalIntent.STOP: {
        LongitudinalExecution.STOPPED: 1.0,
        LongitudinalExecution.DECELERATING: 0.7,
        LongitudinalExecution.CONSTANT_SPEED: 0.0,
        LongitudinalExecution.ACCELERATING: 0.0,
    },
    LongitudinalIntent.SLOW_DOWN: {
        LongitudinalExecution.STOPPED: 0.5,
        LongitudinalExecution.DECELERATING: 1.0,
        LongitudinalExecution.CONSTANT_SPEED: 0.1,
        LongitudinalExecution.ACCELERATING: 0.0,
    },
    LongitudinalIntent.MAINTAIN: {
        LongitudinalExecution.STOPPED: 0.0,
        LongitudinalExecution.DECELERATING: 0.3,
        LongitudinalExecution.CONSTANT_SPEED: 1.0,
        LongitudinalExecution.ACCELERATING: 0.3,
    },
    LongitudinalIntent.ACCELERATE: {
        LongitudinalExecution.STOPPED: 0.0,
        LongitudinalExecution.DECELERATING: 0.0,
        LongitudinalExecution.CONSTANT_SPEED: 0.3,
        LongitudinalExecution.ACCELERATING: 1.0,
    },
}

LATERAL_COMPAT = {
    LateralIntent.NUDGE_LEFT: {
        LateralExecution.SHIFT_LEFT: 1.0,
        LateralExecution.SHIFT_RIGHT: 0.0,
        LateralExecution.LANE_CHANGE_LEFT: 0.6,
        LateralExecution.LANE_CHANGE_RIGHT: 0.0,
        LateralExecution.STRAIGHT: 0.2,
    },
    LateralIntent.NUDGE_RIGHT: {
        LateralExecution.SHIFT_LEFT: 0.0,
        LateralExecution.SHIFT_RIGHT: 1.0,
        LateralExecution.LANE_CHANGE_LEFT: 0.0,
        LateralExecution.LANE_CHANGE_RIGHT: 0.6,
        LateralExecution.STRAIGHT: 0.2,
    },
    LateralIntent.LANE_CHANGE_LEFT: {
        LateralExecution.SHIFT_LEFT: 0.7,
        LateralExecution.SHIFT_RIGHT: 0.0,
        LateralExecution.LANE_CHANGE_LEFT: 1.0,
        LateralExecution.LANE_CHANGE_RIGHT: 0.0,
        LateralExecution.STRAIGHT: 0.0,
    },
    LateralIntent.LANE_CHANGE_RIGHT: {
        LateralExecution.SHIFT_LEFT: 0.0,
        LateralExecution.SHIFT_RIGHT: 0.7,
        LateralExecution.LANE_CHANGE_LEFT: 0.0,
        LateralExecution.LANE_CHANGE_RIGHT: 1.0,
        LateralExecution.STRAIGHT: 0.0,
    },
    LateralIntent.HOLD_LANE: {
        LateralExecution.SHIFT_LEFT: 0.2,
        LateralExecution.SHIFT_RIGHT: 0.2,
        LateralExecution.LANE_CHANGE_LEFT: 0.0,
        LateralExecution.LANE_CHANGE_RIGHT: 0.0,
        LateralExecution.STRAIGHT: 1.0,
    },
}


# ── Text parser ──────────────────────────────────────────────────────

class TextReasoningParser:
    """Parse CoC text into independent longitudinal and lateral intents."""

    def __init__(self):
        self.long_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in LONGITUDINAL_PATTERNS.items()
        }
        self.lat_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in LATERAL_PATTERNS.items()
        }

    def parse(self, text: str) -> tuple[LongitudinalIntent, LateralIntent, float]:
        if not text or not text.strip():
            return LongitudinalIntent.UNSPECIFIED, LateralIntent.UNSPECIFIED, 0.0

        long_counts: dict[LongitudinalIntent, int] = {}
        for intent, patterns in self.long_patterns.items():
            c = sum(len(p.findall(text)) for p in patterns)
            if c > 0:
                long_counts[intent] = c

        lat_counts: dict[LateralIntent, int] = {}
        for intent, patterns in self.lat_patterns.items():
            c = sum(len(p.findall(text)) for p in patterns)
            if c > 0:
                lat_counts[intent] = c

        long_intent = (
            max(long_counts, key=long_counts.get)
            if long_counts else LongitudinalIntent.UNSPECIFIED
        )
        lat_intent = (
            max(lat_counts, key=lat_counts.get)
            if lat_counts else LateralIntent.UNSPECIFIED
        )

        all_counts = list(long_counts.values()) + list(lat_counts.values())
        total = sum(all_counts)
        best = max(all_counts) if all_counts else 0
        confidence = best / total if total > 0 else 0.0

        return long_intent, lat_intent, confidence


# ── Trajectory parser ────────────────────────────────────────────────

class TrajectoryParser:
    """Classify trajectory into longitudinal + lateral behavior."""

    STOPPED_SPEED = 0.5       # m/s
    ACCEL_THRESHOLD = 0.3     # m/s^2
    LATERAL_SHIFT_MIN = 0.2   # meters — even small nudges count
    LANE_CHANGE_MIN = 1.5     # meters — full lane-width shift

    def __init__(self, dt: float = 0.1):
        self.dt = dt

    def parse(self, trajectory: np.ndarray) -> TrajectoryFeatures:
        if trajectory.ndim != 2:
            raise ValueError(f"Need 2-D trajectory, got {trajectory.shape}")

        positions = trajectory[:, :3]
        n = len(positions)

        if n < 2:
            return self._empty()

        diffs = np.diff(positions, axis=0)
        speeds = np.linalg.norm(diffs[:, :2], axis=1) / self.dt

        mean_accel = 0.0
        if len(speeds) > 1:
            mean_accel = float(np.mean(np.diff(speeds) / self.dt))

        lateral_shift = float(positions[-1, 1] - positions[0, 1])
        max_lat_dev = float(np.max(np.abs(positions[:, 1] - positions[0, 1])))
        total_dist = float(np.sum(np.linalg.norm(diffs[:, :2], axis=1)))

        final_speed = float(speeds[-1])
        if final_speed < self.STOPPED_SPEED:
            long_exec = LongitudinalExecution.STOPPED
        elif mean_accel > self.ACCEL_THRESHOLD:
            long_exec = LongitudinalExecution.ACCELERATING
        elif mean_accel < -self.ACCEL_THRESHOLD:
            long_exec = LongitudinalExecution.DECELERATING
        else:
            long_exec = LongitudinalExecution.CONSTANT_SPEED

        abs_lat = abs(lateral_shift)
        if abs_lat > self.LANE_CHANGE_MIN:
            lat_exec = (
                LateralExecution.LANE_CHANGE_LEFT if lateral_shift > 0
                else LateralExecution.LANE_CHANGE_RIGHT
            )
        elif abs_lat > self.LATERAL_SHIFT_MIN:
            lat_exec = (
                LateralExecution.SHIFT_LEFT if lateral_shift > 0
                else LateralExecution.SHIFT_RIGHT
            )
        else:
            lat_exec = LateralExecution.STRAIGHT

        return TrajectoryFeatures(
            initial_speed=float(speeds[0]),
            final_speed=final_speed,
            mean_speed=float(np.mean(speeds)),
            speed_change=float(speeds[-1] - speeds[0]),
            mean_acceleration=mean_accel,
            total_lateral_shift=lateral_shift,
            max_lateral_deviation=max_lat_dev,
            total_distance=total_dist,
            longitudinal=long_exec,
            lateral=lat_exec,
        )

    def _empty(self) -> TrajectoryFeatures:
        return TrajectoryFeatures(
            initial_speed=0.0, final_speed=0.0, mean_speed=0.0,
            speed_change=0.0, mean_acceleration=0.0,
            total_lateral_shift=0.0, max_lateral_deviation=0.0,
            total_distance=0.0,
            longitudinal=LongitudinalExecution.STOPPED,
            lateral=LateralExecution.STRAIGHT,
        )


# ── Scorer ───────────────────────────────────────────────────────────

class MismatchScorer:
    """
    Compare text reasoning to trajectory execution on two independent axes.

    Scoring logic:
      - Each axis (longitudinal, lateral) gets a compatibility score [0,1].
      - Only axes that the text actually specifies contribute to the score.
      - Final mismatch = 1 - weighted_compatibility.
    """

    def __init__(self):
        self.text_parser = TextReasoningParser()
        self.traj_parser = TrajectoryParser()

    def score(self, reasoning_text: str, trajectory: np.ndarray) -> MismatchResult:
        long_intent, lat_intent, confidence = self.text_parser.parse(reasoning_text)
        features = self.traj_parser.parse(trajectory)

        long_specified = long_intent != LongitudinalIntent.UNSPECIFIED
        lat_specified = lat_intent != LateralIntent.UNSPECIFIED

        long_compat = (
            LONGITUDINAL_COMPAT.get(long_intent, {}).get(features.longitudinal, 0.5)
            if long_specified else None
        )
        lat_compat = (
            LATERAL_COMPAT.get(lat_intent, {}).get(features.lateral, 0.5)
            if lat_specified else None
        )

        if long_specified and lat_specified:
            combined = 0.5 * long_compat + 0.5 * lat_compat
        elif long_specified:
            combined = long_compat
        elif lat_specified:
            combined = lat_compat
        else:
            combined = 0.5  # can't assess — no penalty, no reward

        mismatch_score = round(1.0 - combined, 3)

        if not long_specified and not lat_specified:
            mismatch_type = "unclassified_intent"
        elif mismatch_score < 0.3:
            mismatch_type = "consistent"
        elif mismatch_score < 0.6:
            mismatch_type = "partial_mismatch"
        else:
            mismatch_type = "severe_mismatch"

        return MismatchResult(
            longitudinal_intent=long_intent,
            lateral_intent=lat_intent,
            text_confidence=confidence,
            longitudinal_execution=features.longitudinal,
            lateral_execution=features.lateral,
            trajectory_features=features,
            longitudinal_match=long_compat if long_compat is not None else 0.5,
            lateral_match=lat_compat if lat_compat is not None else 0.5,
            mismatch_score=mismatch_score,
            mismatch_type=mismatch_type,
        )


# ── Self-test ────────────────────────────────────────────────────────

def test_scorer():
    scorer = MismatchScorer()

    cases = [
        (
            "Nudge to the left to increase clearance from the construction cones.",
            np.column_stack([
                np.linspace(0, 30, 64),
                np.linspace(0, 0.5, 64),
                np.zeros(64),
            ]),
            "nudge_left + shift_left -> consistent",
        ),
        (
            "Stop for the pedestrian crossing ahead.",
            np.column_stack([
                np.linspace(0, 2, 64),
                np.zeros(64),
                np.zeros(64),
            ]),
            "stop + stopped -> consistent",
        ),
        (
            "Slow down and stay in the current lane.",
            np.column_stack([
                np.linspace(0, 40, 64),
                np.zeros(64),
                np.zeros(64),
            ]),
            "slow_down+hold_lane vs constant_speed+straight -> mismatch",
        ),
        (
            "Accelerate to merge into the left lane.",
            np.column_stack([
                np.cumsum(np.linspace(0.5, 2.0, 64)),
                np.linspace(0, 3.0, 64),
                np.zeros(64),
            ]),
            "accelerate+LC_left vs accel+LC_left -> consistent",
        ),
        (
            "Proceed through the intersection.",
            np.column_stack([
                np.linspace(0, 50, 64),
                np.zeros(64),
                np.zeros(64),
            ]),
            "maintain+unspec_lat vs const+straight -> consistent",
        ),
    ]

    for text, traj, description in cases:
        r = scorer.score(text, traj)
        print(f"{description}")
        print(f"  Long: {r.longitudinal_intent.value} vs "
              f"{r.longitudinal_execution.value} -> {r.longitudinal_match:.2f}")
        print(f"  Lat:  {r.lateral_intent.value} vs "
              f"{r.lateral_execution.value} -> {r.lateral_match:.2f}")
        print(f"  Score: {r.mismatch_score:.3f} ({r.mismatch_type})")
        print()


if __name__ == "__main__":
    test_scorer()
