"""Metric calculators, cross-checked against py-motmetrics where a reference
implementation exists, and against closed forms where one doesn't."""

import math

import motmetrics as mm
import numpy as np
import pytest

from vision.alerts import Alarm
from vision.analytics.gait import gait_asymmetry_index, stance_durations
from vision.eval.manifest import FRAME, TRACK_ID, GroundTruthEvent, SceneParams
from vision.eval.metrics import (
    LatencyReport,
    clear_mot,
    identity_counts,
    match_events,
    miss_breakdown,
    poisson_rate_upper,
)
from vision.synth.scene import generate_scene


def _gt() -> np.ndarray:
    return generate_scene(SceneParams(duration_s=10, n_targets=4, occluder=False), seed=7).gt_boxes


def _corrupt(gt: np.ndarray, seed: int) -> np.ndarray:
    """A plausible bad tracker: jittered boxes, dropped detections, false
    positives, and an identity swap between two tracks halfway through."""
    rng = np.random.default_rng(seed)
    hyp = gt.copy()
    hyp[:, 2:6] += rng.normal(0, 2.0, (len(hyp), 4))
    hyp = hyp[rng.random(len(hyp)) > 0.1]
    half = hyp[:, FRAME] > hyp[:, FRAME].max() / 2
    a, b = hyp[:, TRACK_ID] == 1, hyp[:, TRACK_ID] == 2
    hyp[half & a, TRACK_ID], hyp[half & b, TRACK_ID] = 2, 1
    hyp[:, TRACK_ID] += 100  # hypothesis ids are the tracker's own, never the gt's
    n_fp = 60
    fp = np.column_stack(
        [
            rng.integers(1, int(gt[:, FRAME].max()) + 1, n_fp),
            1000 + np.arange(n_fp),
            rng.uniform(0, 300, n_fp),
            rng.uniform(0, 220, n_fp),
            np.full(n_fp, 20),
            np.full(n_fp, 20),
            np.ones(n_fp),
        ]
    )
    return np.vstack([hyp, fp])


@pytest.fixture(autouse=True)
def _numpy2_shim_for_motmetrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """py-motmetrics 1.4.0 (latest on PyPI) still calls np.asfarray, removed in
    NumPy 2.0. Restoring exactly that function, test-only, keeps the cross-check
    running motmetrics' own IoU code rather than substituting ours."""
    monkeypatch.setattr(np, "asfarray", lambda a, dtype=np.float64: np.asarray(a, dtype=dtype), raising=False)


def _motmetrics(gt: np.ndarray, hyp: np.ndarray) -> dict[str, float]:
    acc = mm.MOTAccumulator(auto_id=False)
    for f in np.unique(np.concatenate([gt[:, FRAME], hyp[:, FRAME]])):
        g, h = gt[gt[:, FRAME] == f], hyp[hyp[:, FRAME] == f]
        dist = mm.distances.iou_matrix(g[:, 2:6], h[:, 2:6], max_iou=0.5)
        acc.update(g[:, TRACK_ID].astype(int), h[:, TRACK_ID].astype(int), dist, frameid=int(f))
    row = mm.metrics.create().compute(
        acc, metrics=["mota", "motp", "idf1", "num_switches", "num_false_positives", "num_misses"]
    )
    return {k: float(v) for k, v in row.iloc[0].items()}


@pytest.mark.parametrize("seed", range(5))
def test_matches_py_motmetrics(seed: int) -> None:
    gt = _gt()
    hyp = _corrupt(gt, seed)
    ours, ident, ref = clear_mot(gt, hyp), identity_counts(gt, hyp), _motmetrics(gt, hyp)
    # Agreement only means something if the fixture exercises every error type.
    assert ours.misses > 0
    assert ours.false_positives > 0
    assert ours.id_switches > 0
    assert ours.id_switches == ref["num_switches"]
    assert ours.false_positives == ref["num_false_positives"]
    assert ours.misses == ref["num_misses"]
    assert ours.mota == pytest.approx(ref["mota"], abs=1e-9)
    assert ours.motp == pytest.approx(ref["motp"], abs=1e-9)
    assert ident.idf1 == pytest.approx(ref["idf1"], abs=1e-9)


def test_perfect_tracker() -> None:
    gt = _gt()
    c, i = clear_mot(gt, gt), identity_counts(gt, gt)
    assert (c.mota, i.idf1, c.id_switches) == (1.0, 1.0, 0)
    assert c.motp == pytest.approx(0.0, abs=1e-12)


def test_id_swap_hits_idf1_far_harder_than_mota() -> None:
    gt = _gt()
    hyp = gt.copy()
    half = hyp[:, FRAME] > hyp[:, FRAME].max() / 2
    a, b = hyp[:, TRACK_ID] == 1, hyp[:, TRACK_ID] == 2
    hyp[half & a, TRACK_ID], hyp[half & b, TRACK_ID] = 2, 1
    c, i = clear_mot(gt, hyp), identity_counts(gt, hyp)
    assert c.id_switches == 2
    assert c.mota == pytest.approx(1 - 2 / len(gt))
    assert i.idf1 < 0.85  # two of four identities are half wrong


def test_negative_control_random_tracker_scores_badly() -> None:
    gt = _gt()
    rng = np.random.default_rng(0)
    hyp = gt.copy()
    hyp[:, 2:4] = rng.uniform(0, 300, (len(hyp), 2))
    assert clear_mot(gt, hyp).mota < 0
    assert identity_counts(gt, hyp).idf1 < 0.1


def test_empty_hypothesis_is_all_misses() -> None:
    gt = _gt()
    c = clear_mot(gt, np.empty((0, 7)))
    assert (c.mota, c.misses, c.false_positives) == (0.0, len(gt), 0)
    assert identity_counts(gt, np.empty((0, 7))).idf1 == 0.0


def test_poisson_upper_is_rule_of_three() -> None:
    assert poisson_rate_upper(0, 1.0) == pytest.approx(-math.log(0.05))  # 2.9957
    assert poisson_rate_upper(0, 3.0) < 1.0 < poisson_rate_upper(0, 2.9)
    # More events, same exposure: the bound must rise.
    assert poisson_rate_upper(5, 3.0) > poisson_rate_upper(1, 3.0)


def test_event_matching() -> None:
    events = [GroundTruthEvent(kind="surge", start_s=10, end_s=12)]
    alarms = [
        Alarm("surge", 10.5, 11, 1),
        Alarm("surge", 11.0, 12, 2),  # second alarm on the same event: not a false alarm
        Alarm("surge", 50.0, 51, 3),  # nowhere near an event
        Alarm("intrusion", 10.5, 11, 4),  # right time, wrong kind
    ]
    c = match_events(alarms, events, exposure_s=86_400)
    assert (c.events, c.detected, c.false_alarms, c.recall, c.facd) == (1, 1, 2, 1.0, 2.0)


def test_latency_report() -> None:
    r = LatencyReport.from_ns([i * 1_000_000 for i in range(1, 101)])
    assert r.n == 100
    assert r.max_ms == 100.0
    assert r.p50_ci95_ms[0] <= r.p50_ms <= r.p50_ci95_ms[1]
    assert r.p50_ms < r.p95_ms < r.p99_ms <= r.max_ms


def test_gai_and_stance() -> None:
    assert gait_asymmetry_index(0.6, 0.6) == 0.0
    assert gait_asymmetry_index(0.5, 0.7) == pytest.approx(100 * 0.2 / 0.6)
    contacts = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1], dtype=bool)
    # Runs touching either end are partial and dropped.
    assert stance_durations(contacts, fps=10) == pytest.approx([0.3, 0.2])


def test_duplicate_id_in_a_frame_is_rejected() -> None:
    gt = np.array([[f, 1, 0, 0, 10, 10, 1] for f in (1, 2, 3)], dtype=float)
    dup = np.vstack([gt, gt])  # every (frame, id) twice: IDF1 would come out at 4/3
    dup[:, TRACK_ID] = 7
    with pytest.raises(ValueError, match="more than once"):
        identity_counts(gt, dup)
    with pytest.raises(ValueError, match="more than once"):
        clear_mot(gt, dup)
    assert identity_counts(gt, dup[:3]).idf1 == 1.0  # control: the same rows, once each, are fine


def test_degenerate_inputs_do_not_crash() -> None:
    gt = _gt()
    assert identity_counts(np.empty((0,)), gt).idf1 == 0.0
    with pytest.raises(ValueError, match="box array"):
        clear_mot(gt[:, :3], gt)
    c = match_events([], [], exposure_s=0.0)
    assert math.isnan(c.facd)
    assert math.isnan(c.facd_upper95)


def test_track_matched_events_do_not_credit_the_wrong_track() -> None:
    events = [GroundTruthEvent(kind="surge", start_s=10, end_s=12, track_ids=(2,))]
    alarm_on_other_track = [Alarm("surge", 10.5, 11, 1)]
    loose = match_events(alarm_on_other_track, events, 86_400)
    strict = match_events(alarm_on_other_track, events, 86_400, match_track_ids=True)
    assert (loose.detected, loose.false_alarms) == (1, 0)
    assert (strict.detected, strict.false_alarms) == (0, 1)
    right_track = match_events([Alarm("surge", 10.5, 11, 2)], events, 86_400, match_track_ids=True)
    assert (right_track.detected, right_track.false_alarms) == (1, 0)


def test_miss_breakdown_splits_unseen_from_badly_boxed() -> None:
    gt = np.array([[1, 1, 0, 0, 10, 10, 1], [1, 2, 50, 50, 10, 10, 1]], dtype=float)
    hyp = np.array([[1, 9, 5, 0, 10, 10, 1]], dtype=float)  # half-overlaps gt 1, misses gt 2
    assert miss_breakdown(gt, hyp) == (1, 1)
    assert miss_breakdown(gt, gt) == (0, 0)
