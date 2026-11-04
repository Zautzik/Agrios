"""The evaluation gate: run every sequence in a manifest, score it, compare the
suite totals with the manifest's criteria, write report.json + summary.md,
and exit nonzero if any criterion fails.

    uv run python -m vision.eval.gate --manifest vision/eval/suites/phase1.json --out vision-results
    uv run python -m vision.eval.gate ... --control noise --expect-fail
    uv run python -m vision.eval.gate ... --control shuffle --expect-fail

The two controls keep every sequence's ground truth but corrupt its input:
`noise` replaces it with signal-free noise, `shuffle` permutes its time
order. A gate that still passed under either would be passing on something
other than the temporal structure it claims to measure -- so CI runs both and
requires them to fail, the same way the C++ side requires its sanitizer
negative controls to fail. --expect-fail inverts the exit code and tightens it:
success means *every* suite failed at least one check, not just the gate as a
whole, so one suite that happens to catch the corruption can't cover for
another that doesn't.

A gate with nothing to check doesn't pass. A criteria entry with no sequences
behind it, a suite with no criteria, or a run with no checks at all is a
failure: an empty `all()` is True, and "passed" must never mean "checked
nothing".
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from vision.analytics.gait import gait_asymmetry_index, stance_durations
from vision.eval.manifest import (
    GaitSource,
    GateCriteria,
    KinematicsSource,
    Manifest,
    SceneSource,
)
from vision.eval.metrics import (
    SECONDS_PER_DAY,
    ClearMotCounts,
    EventCounts,
    IdentityCounts,
    LatencyReport,
    clear_mot,
    gai_error,
    identity_counts,
    machine_stamp,
    match_events,
    miss_breakdown,
)
from vision.reference.pipeline import run_pixel_pipeline, surge_alarms_from_tracks
from vision.synth import LIMBS
from vision.synth.gait import generate_gait
from vision.synth.kinematics import generate_kinematics
from vision.synth.scene import generate_scene

Control = Literal["none", "noise", "shuffle"]
SuiteKind = Literal["scene", "kinematics", "gait", "file"]
_CONTROL_SEED = 0xA9105  # fixed, so a control run is as reproducible as a normal one


def _control_rng(seed: int) -> np.random.Generator:
    # Mixed with the sequence's own seed, so each sequence's corruption is
    # independent: one shared stream would corrupt equal-shaped sequences
    # identically and make them one sample, not several.
    return np.random.default_rng([_CONTROL_SEED, seed])


Metrics = dict[str, float | int]


class Check(BaseModel):
    metric: str
    value: float | None
    bound: float
    kind: Literal["min", "max"]
    passed: bool


class SuiteReport(BaseModel):
    sequences: int
    metrics: Metrics
    frame_latency: LatencyReport | None = None
    checks: list[Check]


class GateReport(BaseModel):
    control: Control
    passed: bool
    machine: dict[str, str | int | None]
    suites: dict[SuiteKind, SuiteReport]
    sequences: dict[str, Metrics]


@dataclass
class _Totals:
    """Raw counts summed over a suite's sequences, so suite metrics are pooled
    (as py-motmetrics' OVERALL row is), not averages of per-sequence ratios."""

    sequences: int = 0
    clear: ClearMotCounts | None = None
    identity: IdentityCounts | None = None
    events: EventCounts | None = None
    gai_errors: list[float] = field(default_factory=list)
    latency_ns: list[int] = field(default_factory=list)


def _add[T: (ClearMotCounts, IdentityCounts, EventCounts)](a: T | None, b: T) -> T:
    return b if a is None else a + b


def _run_scene(src: SceneSource, control: Control, totals: _Totals) -> Metrics:
    seq = generate_scene(src.params, src.seed)
    frames = seq.frames
    rng = _control_rng(src.seed)
    if control == "noise":
        frames = rng.integers(0, 256, frames.shape, dtype=np.uint8)
    elif control == "shuffle":
        frames = frames[rng.permutation(len(frames))]
    result = run_pixel_pipeline(frames, seq.fps)
    clear = clear_mot(seq.gt_boxes, result.hyp_boxes)
    ident = identity_counts(seq.gt_boxes, result.hyp_boxes)
    events = match_events(result.alarms, seq.events, seq.duration_s)
    misses_no_overlap, misses_partial = miss_breakdown(seq.gt_boxes, result.hyp_boxes)
    totals.clear = _add(totals.clear, clear)
    totals.identity = _add(totals.identity, ident)
    totals.events = _add(totals.events, events)
    totals.latency_ns += result.frame_latency_ns
    return {
        "mota": clear.mota,
        "idf1": ident.idf1,
        "id_switches": clear.id_switches,
        "misses": clear.misses,
        "misses_no_overlap": misses_no_overlap,
        "misses_partial_overlap": misses_partial,
        "false_positives": clear.false_positives,
        "events": events.events,
        "detected": events.detected,
        "false_alarms": events.false_alarms,
        "scene_changes_suppressed": result.scene_changes,
    }


def _run_kinematics(src: KinematicsSource, control: Control, totals: _Totals) -> Metrics:
    seq = generate_kinematics(src.params, src.seed)
    tracks = seq.tracks.copy()
    rng = _control_rng(src.seed)
    if control == "noise":
        tracks[:, 2:4] = rng.uniform([0, 0], src.params.arena_m, (len(tracks), 2))
    elif control == "shuffle":
        for tid in np.unique(tracks[:, 1]):
            idx = np.flatnonzero(tracks[:, 1] == tid)
            tracks[idx, 2:4] = tracks[rng.permutation(idx), 2:4]
    alarms = surge_alarms_from_tracks(tracks, seq.fps)
    # Alarm ids here *are* ground-truth ids, so an alarm only counts for its own track's event.
    events = match_events(alarms, seq.events, seq.duration_s, match_track_ids=True)
    totals.events = _add(totals.events, events)
    return {
        "camera_days": seq.duration_s / SECONDS_PER_DAY,
        "events": events.events,
        "detected": events.detected,
        "false_alarms": events.false_alarms,
    }


def _run_gait(src: GaitSource, control: Control, totals: _Totals) -> Metrics:
    seq = generate_gait(src.params, src.seed)
    contacts = seq.contacts
    rng = _control_rng(src.seed)
    if control == "noise":
        contacts = rng.random(contacts.shape) < src.params.duty_factor
    elif control == "shuffle":
        contacts = contacts[rng.permutation(len(contacts))]
    left, right = seq.pair
    stances = {limb: stance_durations(contacts[:, LIMBS.index(limb)], seq.fps) for limb in (left, right)}
    if not stances[left] or not stances[right]:
        err = float("inf")  # no complete stance found: as wrong as an estimate gets
    else:
        est = gait_asymmetry_index(float(np.mean(stances[left])), float(np.mean(stances[right])))
        err = gai_error(est, seq.true_gai)
    totals.gai_errors.append(err)
    return {"true_gai": seq.true_gai, "gai_error": err}


def _metrics(t: _Totals) -> Metrics:
    m: Metrics = {}
    if t.clear:
        m |= {"mota": t.clear.mota, "motp": t.clear.motp, "id_switches": t.clear.id_switches}
    if t.identity:
        m["idf1"] = t.identity.idf1
    if t.events:
        e = t.events
        m |= {
            "events": e.events,
            "detected": e.detected,
            "event_recall": e.recall,
            "false_alarms": e.false_alarms,
            "camera_days": e.exposure_s / SECONDS_PER_DAY,
            "facd": e.facd,
            "facd_upper95": e.facd_upper95,
        }
    if t.gai_errors:
        m["max_gai_error"] = max(t.gai_errors)
    return m


# Which GateCriteria field bounds which metric, and in which direction.
_BOUNDS: tuple[tuple[str, str, Literal["min", "max"]], ...] = (
    ("event_recall", "min_event_recall", "min"),
    ("false_alarms", "max_false_alarms", "max"),
    ("facd_upper95", "max_facd_upper95", "max"),
    ("mota", "min_mota", "min"),
    ("idf1", "min_idf1", "min"),
    ("max_gai_error", "max_gai_error", "max"),
)


def _checks(criteria: GateCriteria, metrics: Metrics) -> list[Check]:
    checks = []
    for metric, criterion, kind in _BOUNDS:
        bound: float | None = getattr(criteria, criterion)
        if bound is None:
            continue
        value = metrics.get(metric)
        ok = value is not None and not np.isnan(value)
        if ok and value is not None:
            ok = value >= bound if kind == "min" else value <= bound
        checks.append(Check(metric=metric, value=value, bound=bound, kind=kind, passed=ok))
    return checks


def run_gate(manifest: Manifest, control: Control = "none") -> GateReport:
    totals: dict[SuiteKind, _Totals] = {}
    per_sequence: dict[str, Metrics] = {}
    for spec in manifest.sequences:
        src = spec.source
        if isinstance(src, SceneSource):
            t = totals.setdefault("scene", _Totals())
            per_sequence[spec.id] = _run_scene(src, control, t)
        elif isinstance(src, KinematicsSource):
            t = totals.setdefault("kinematics", _Totals())
            per_sequence[spec.id] = _run_kinematics(src, control, t)
        elif isinstance(src, GaitSource):
            t = totals.setdefault("gait", _Totals())
            per_sequence[spec.id] = _run_gait(src, control, t)
        else:
            raise NotImplementedError(f"{spec.id}: recorded footage is read by vision.sources (Phase 2)")
        t.sequences += 1

    suites: dict[SuiteKind, SuiteReport] = {}
    for kind, t in totals.items():
        metrics = _metrics(t)
        criteria = manifest.criteria.get(kind)
        checks = _checks(criteria, metrics) if criteria else []
        if not checks:
            checks = [Check(metric="criteria_defined", value=0, bound=1, kind="min", passed=False)]
        suites[kind] = SuiteReport(
            sequences=t.sequences,
            metrics=metrics,
            frame_latency=LatencyReport.from_ns(t.latency_ns) if t.latency_ns else None,
            checks=checks,
        )
    for kind in manifest.criteria.keys() - suites.keys():
        # Criteria for a suite that never ran: its bounds would otherwise vanish silently.
        suites[kind] = SuiteReport(
            sequences=0,
            metrics={},
            checks=[Check(metric="sequences", value=0, bound=1, kind="min", passed=False)],
        )
    all_checks = [c for s in suites.values() for c in s.checks]
    passed = bool(all_checks) and all(c.passed for c in all_checks)
    return GateReport(
        control=control, passed=passed, machine=machine_stamp(), suites=suites, sequences=per_sequence
    )


def suites_not_failing(report: GateReport) -> list[str]:
    """Suites in which every check passed -- under a negative control, the
    suites that failed to notice the corruption."""
    return [k for k, s in report.suites.items() if all(c.passed for c in s.checks)]


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def render_summary(report: GateReport) -> str:
    lines = [
        f"## Vision gate: {'PASS' if report.passed else 'FAIL'} (control: {report.control})",
        "",
        "Reference pipeline on synthetic fixtures: this validates the harness, not model performance.",
        "",
        "| suite | metric | value | bound | result |",
        "|---|---|---|---|---|",
    ]
    for kind, s in report.suites.items():
        for c in s.checks:
            sign = ">=" if c.kind == "min" else "<="
            result = "pass" if c.passed else "**FAIL**"
            lines.append(f"| {kind} | {c.metric} | {_fmt(c.value)} | {sign} {_fmt(c.bound)} | {result} |")
        if lat := s.frame_latency:
            lines.append(
                f"| {kind} | frame latency ms (n={lat.n}) | p50 {lat.p50_ms:.3f} "
                f"[95% CI {lat.p50_ci95_ms[0]:.3f}-{lat.p50_ci95_ms[1]:.3f}], "
                f"p95 {lat.p95_ms:.3f}, p99 {lat.p99_ms:.3f}, max {lat.max_ms:.3f} | - | info |"
            )
    m = report.machine
    lines += [
        "",
        f"Machine: {m['cpu']}, {m['logical_cpus']} logical CPUs, {m['os']}, "
        f"Python {m['python']}, NumPy {m['numpy']}, OpenCV {m['opencv']}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the vision evaluation gate.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control", choices=["none", "noise", "shuffle"], default="none")
    parser.add_argument(
        "--expect-fail",
        action="store_true",
        help="exit 0 only if every suite fails at least one check (for negative controls)",
    )
    args = parser.parse_args(argv)

    report = run_gate(Manifest.load(args.manifest), args.control)
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.control == "none" else f"-{args.control}"
    (args.out / f"report{suffix}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    summary = render_summary(report)
    (args.out / f"summary{suffix}.md").write_text(summary, encoding="utf-8")
    print(summary)
    if args.expect_fail:
        missed = suites_not_failing(report)
        if missed:
            print(f"negative control NOT caught by: {', '.join(missed)}")
            return 1
        print(f"negative control caught by every suite: {', '.join(report.suites)}")
        return 0
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
