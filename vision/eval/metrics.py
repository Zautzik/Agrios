"""Metric calculators: CLEAR-MOT (MOTA/MOTP), IDF1, event recall and False
Alarms per Camera-Day (FACD), GAI error, and latency percentiles.

MOTA and IDF1 are implemented here, not just imported from py-motmetrics,
because the gate has to run them on every sequence in CI and their exact
matching rules are part of what's being tested. The test suite checks both
against py-motmetrics on fixtures that exercise every error type (misses,
false positives, ID switches), so "matches the reference implementation" is
a measured property, not an assumption.

Matching rules, as in py-motmetrics / the CLEAR-MOT paper (Bernardin &
Stiefelhagen 2008):
  - a ground-truth/hypothesis pair can match only if IoU >= iou_threshold;
  - a correspondence from an earlier frame is kept while it stays valid, and
    only the rest is solved by Hungarian assignment on 1 - IoU;
  - an ID switch is a ground-truth object matched to a different hypothesis
    than the one it was *last* matched to, however many frames ago.
IDF1 (Ristani et al. 2016) is the global one-to-one ID assignment that
maximises identity true positives over the whole sequence.
"""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from vision.alerts import Alarm
from vision.eval.manifest import FRAME, HEIGHT, LEFT, TOP, TRACK_ID, WIDTH, Boxes, GroundTruthEvent

SECONDS_PER_DAY = 86_400.0


def iou_matrix(a: Boxes, b: Boxes) -> npt.NDArray[np.float64]:
    """Pairwise IoU of two box arrays (rows in vision.eval.manifest layout)."""
    ax0, ay0 = a[:, LEFT, None], a[:, TOP, None]
    ax1, ay1 = ax0 + a[:, WIDTH, None], ay0 + a[:, HEIGHT, None]
    bx0, by0 = b[None, :, LEFT], b[None, :, TOP]
    bx1, by1 = bx0 + b[None, :, WIDTH], by0 + b[None, :, HEIGHT]
    iw = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)
    ih = np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)
    inter = iw * ih
    union = a[:, WIDTH, None] * a[:, HEIGHT, None] + b[None, :, WIDTH] * b[None, :, HEIGHT] - inter
    out: npt.NDArray[np.float64] = np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)
    return out


def _validated(boxes: Boxes, name: str) -> Boxes:
    """Box arrays as every metric here expects them: 2-D with at least the six
    MOTChallenge columns, and at most one row per (frame, id). A duplicated id
    in one frame isn't a tracker quirk to tolerate -- it makes IDF1 exceed 1 --
    so it's rejected, for recorded ground truth and tracker output alike."""
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 7))
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"{name}: need an (N, >=6) box array, got shape {arr.shape}")
    keys = arr[:, [FRAME, TRACK_ID]].astype(np.int64)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError(f"{name}: an id appears more than once in the same frame")
    return arr


def _by_frame(boxes: Boxes) -> dict[int, Boxes]:
    if len(boxes) == 0:  # np.split of an empty array yields one chunk, not zero
        return {}
    order = np.argsort(boxes[:, FRAME], kind="stable")
    boxes = boxes[order]
    frames, starts = np.unique(boxes[:, FRAME].astype(np.int64), return_index=True)
    return dict(zip(frames.tolist(), np.split(boxes, starts[1:]), strict=True))


# ---- CLEAR-MOT ----


@dataclass(frozen=True, slots=True)
class ClearMotCounts:
    """Raw counts, so several sequences aggregate by summing (as py-motmetrics'
    OVERALL row does) instead of averaging per-sequence ratios."""

    num_gt: int
    matches: int
    false_positives: int
    misses: int
    id_switches: int
    sum_distance: float  # sum of (1 - IoU) over matches

    @property
    def mota(self) -> float:
        if self.num_gt == 0:
            return float("nan")
        return 1.0 - (self.misses + self.false_positives + self.id_switches) / self.num_gt

    @property
    def motp(self) -> float:
        """Mean 1 - IoU over matches (py-motmetrics' definition: lower is better)."""
        return self.sum_distance / self.matches if self.matches else float("nan")

    def __add__(self, other: ClearMotCounts) -> ClearMotCounts:
        return ClearMotCounts(
            *(x + y for x, y in zip(asdict(self).values(), asdict(other).values(), strict=True))
        )


def clear_mot(gt: Boxes, hyp: Boxes, iou_threshold: float = 0.5) -> ClearMotCounts:
    gt, hyp = _validated(gt, "ground truth"), _validated(hyp, "hypothesis")
    gt_frames, hyp_frames = _by_frame(gt), _by_frame(hyp)
    empty = np.empty((0, 7))
    last_match: dict[int, int] = {}  # gt id -> hyp id it was last matched to
    num_gt = matches = fps_ = misses = switches = 0
    sum_dist = 0.0
    for f in sorted(gt_frames.keys() | hyp_frames.keys()):
        g, h = gt_frames.get(f, empty), hyp_frames.get(f, empty)
        gids, hids = g[:, TRACK_ID].astype(int).tolist(), h[:, TRACK_ID].astype(int).tolist()
        iou = iou_matrix(g, h) if len(g) and len(h) else np.zeros((len(g), len(h)))
        valid = iou >= iou_threshold
        pairs: list[tuple[int, int]] = []
        used_g: set[int] = set()
        used_h: set[int] = set()
        hid_col = {hid: j for j, hid in enumerate(hids)}
        # 1. Keep last frame's correspondences that are still valid.
        for i, gid in enumerate(gids):
            j = hid_col.get(last_match.get(gid, -1))
            if j is not None and valid[i, j] and j not in used_h:
                pairs.append((i, j))
                used_g.add(i)
                used_h.add(j)
        # 2. Hungarian on the rest; invalid pairs can never be chosen.
        rest_g = [i for i in range(len(gids)) if i not in used_g]
        rest_h = [j for j in range(len(hids)) if j not in used_h]
        if rest_g and rest_h:
            sub_valid = valid[np.ix_(rest_g, rest_h)]
            cost = np.where(sub_valid, 1.0 - iou[np.ix_(rest_g, rest_h)], 1e6)
            for r, c in zip(*linear_sum_assignment(cost), strict=True):
                if sub_valid[r, c]:
                    i, j = rest_g[r], rest_h[c]
                    if gids[i] in last_match and last_match[gids[i]] != hids[j]:
                        switches += 1
                    pairs.append((i, j))
        for i, j in pairs:
            last_match[gids[i]] = hids[j]
            sum_dist += 1.0 - float(iou[i, j])
        num_gt += len(gids)
        matches += len(pairs)
        misses += len(gids) - len(pairs)
        fps_ += len(hids) - len(pairs)
    return ClearMotCounts(num_gt, matches, fps_, misses, switches, sum_dist)


# ---- IDF1 ----


@dataclass(frozen=True, slots=True)
class IdentityCounts:
    idtp: int
    num_gt: int
    num_hyp: int

    @property
    def idf1(self) -> float:
        total = self.num_gt + self.num_hyp
        return 2.0 * self.idtp / total if total else float("nan")

    def __add__(self, other: IdentityCounts) -> IdentityCounts:
        return IdentityCounts(
            self.idtp + other.idtp, self.num_gt + other.num_gt, self.num_hyp + other.num_hyp
        )


def identity_counts(gt: Boxes, hyp: Boxes, iou_threshold: float = 0.5) -> IdentityCounts:
    """IDTP under the best global one-to-one gt-id <-> hyp-id assignment.

    Minimising IDFN + IDFP over that assignment (the paper's formulation) is
    the same as maximising the summed per-pair co-occurrence counts, since
    each pair's cost is n_gt + n_hyp - 2 * overlap -- so that's what is solved.
    """
    gt, hyp = _validated(gt, "ground truth"), _validated(hyp, "hypothesis")
    gt_ids, gt_idx = np.unique(gt[:, TRACK_ID].astype(np.int64), return_inverse=True)
    hyp_ids, hyp_idx = np.unique(hyp[:, TRACK_ID].astype(np.int64), return_inverse=True)
    overlap = np.zeros((len(gt_ids), len(hyp_ids)), dtype=np.int64)
    gt_frames, hyp_frames = (
        _by_frame(np.column_stack([gt, gt_idx])),
        _by_frame(np.column_stack([hyp, hyp_idx])),
    )
    for f, g in gt_frames.items():
        h = hyp_frames.get(f)
        if h is None:
            continue
        valid = iou_matrix(g, h) >= iou_threshold
        gi, hj = np.nonzero(valid)
        np.add.at(overlap, (g[gi, -1].astype(np.int64), h[hj, -1].astype(np.int64)), 1)
    idtp = 0
    if overlap.size:
        rows, cols = linear_sum_assignment(overlap, maximize=True)
        idtp = int(overlap[rows, cols].sum())
    return IdentityCounts(idtp, len(gt), len(hyp))


def miss_breakdown(gt: Boxes, hyp: Boxes, iou_threshold: float = 0.5) -> tuple[int, int]:
    """Ground-truth boxes no hypothesis covers at iou_threshold, split into
    (no overlap at all, partial overlap). Diagnostic, not a metric: it tells
    "the detector never saw it" apart from "it saw it, but the box was wrong"
    (a merged blob, a box clipped by an occluder)."""
    gt, hyp = _validated(gt, "ground truth"), _validated(hyp, "hypothesis")
    hyp_frames = _by_frame(hyp)
    none = partial = 0
    for f, g in _by_frame(gt).items():
        h = hyp_frames.get(f)
        best = iou_matrix(g, h).max(axis=1) if h is not None else np.zeros(len(g))
        none += int(np.count_nonzero(best == 0))
        partial += int(np.count_nonzero((best > 0) & (best < iou_threshold)))
    return none, partial


# ---- Events and false alarms ----


@dataclass(frozen=True, slots=True)
class EventCounts:
    events: int
    detected: int
    false_alarms: int
    exposure_s: float  # camera-seconds observed

    @property
    def recall(self) -> float:
        return self.detected / self.events if self.events else float("nan")

    @property
    def facd(self) -> float:
        if self.exposure_s <= 0:
            return float("nan")  # no observation time: no rate, rather than a crash
        return self.false_alarms / (self.exposure_s / SECONDS_PER_DAY)

    @property
    def facd_upper95(self) -> float:
        if self.exposure_s <= 0:
            return float("nan")
        return poisson_rate_upper(self.false_alarms, self.exposure_s / SECONDS_PER_DAY)

    def __add__(self, other: EventCounts) -> EventCounts:
        return EventCounts(
            self.events + other.events,
            self.detected + other.detected,
            self.false_alarms + other.false_alarms,
            self.exposure_s + other.exposure_s,
        )


def match_events(
    alarms: Sequence[Alarm],
    events: Sequence[GroundTruthEvent],
    exposure_s: float,
    tolerance_s: float = 1.0,
    match_track_ids: bool = False,
) -> EventCounts:
    """An alarm counts toward an event of the same kind if it *starts* inside
    the event window widened by tolerance_s; an alarm matching no event is a
    false alarm. Several alarms on one event (e.g. both tracks of a fight)
    detect it once and aren't false alarms.

    match_track_ids: also require the alarm's track to be one of the event's.
    Use it wherever alarm ids *are* ground-truth ids (the kinematic suites run
    on ground-truth tracks): otherwise an alarm on track A during track B's
    surge would credit B's event and hide A's false alarm. A pixel pipeline's
    tracker ids are its own, so there it stays off."""
    detected = [False] * len(events)
    false_alarms = 0
    for a in alarms:
        hit = False
        for k, e in enumerate(events):
            in_window = e.start_s - tolerance_s <= a.start_s <= e.end_s + tolerance_s
            track_ok = not match_track_ids or a.track_id in e.track_ids
            if a.kind == e.kind and in_window and track_ok:
                detected[k] = hit = True
        false_alarms += not hit
    return EventCounts(len(events), sum(detected), false_alarms, exposure_s)


def poisson_rate_upper(count: int, exposure: float, confidence: float = 0.95) -> float:
    """One-sided exact (Garwood) upper confidence bound on a Poisson rate.

    The one-sided bound is the one a "FACD <= X" claim needs. With zero events
    it reduces to the rule of three: -ln(0.05) / exposure ~= 3 / exposure, so
    ~3 alarm-free camera-days are needed before FACD <= 1 is supported at 95%.
    """
    if exposure <= 0:
        raise ValueError("exposure must be positive")
    return float(chi2.ppf(confidence, 2 * count + 2) / 2.0 / exposure)


# ---- Gait ----


def gai_error(predicted: float, true: float) -> float:
    """Absolute GAI error, in GAI points."""
    return abs(predicted - true)


# ---- Latency ----


@dataclass(frozen=True, slots=True)
class LatencyReport:
    n: int
    p50_ms: float
    p50_ci95_ms: tuple[float, float]
    p95_ms: float
    p99_ms: float
    max_ms: float

    @classmethod
    def from_ns(cls, samples_ns: Iterable[int], seed: int = 0) -> LatencyReport:
        """p50's CI is a percentile bootstrap (2000 resamples, seeded so the
        report is reproducible from the same samples)."""
        ms = np.asarray(list(samples_ns), dtype=np.float64) / 1e6
        if ms.size == 0:
            raise ValueError("no latency samples")
        boots = np.median(np.random.default_rng(seed).choice(ms, (2000, ms.size)), axis=1)
        p50, p95, p99 = np.percentile(ms, [50, 95, 99])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return cls(int(ms.size), float(p50), (float(lo), float(hi)), float(p95), float(p99), float(ms.max()))


def _cpu_model() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith(("model name", "hardware", "model\t")):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def machine_stamp() -> dict[str, str | int | None]:
    """What a latency number is only meaningful next to."""
    import cv2

    return {
        "cpu": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
