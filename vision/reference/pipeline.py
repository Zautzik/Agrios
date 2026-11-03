"""Background subtraction -> blob boxes -> constant-velocity tracker ->
velocity-surge alert. See vision/reference/__init__.py for what this is (and
isn't) for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from vision.alerts import Alarm, AlertConfig, Hysteresis, hysteresis_intervals
from vision.eval.manifest import Boxes

Detections = npt.NDArray[np.float64]  # (K, 4): left, top, width, height


class BlobDetector:
    """MOG2 background subtraction, cleaned up morphologically, one box per
    connected component.

    A frame where most of the image changes at once (an illumination step,
    the IR switch) yields no detections: that's a scene change, not motion,
    and passing its giant blob downstream would read as an object teleporting
    across the frame. Phase 2's tamper/quality detectors replace this guard.
    """

    def __init__(
        self, min_area: int = 80, var_threshold: float = 25.0, scene_change_fraction: float = 0.25
    ) -> None:
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=var_threshold, detectShadows=False
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.min_area = min_area
        self.scene_change_fraction = scene_change_fraction
        self.scene_changes = 0

    def detect(self, frame: npt.NDArray[np.uint8]) -> Detections:
        mask = self._mog.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)
        if np.count_nonzero(mask) > self.scene_change_fraction * mask.size:
            self.scene_changes += 1
            return np.empty((0, 4))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        keep = stats[1:n, cv2.CC_STAT_AREA] >= self.min_area
        return stats[1:n][keep, :4].astype(np.float64)


@dataclass
class _Track:
    track_id: int
    box: npt.NDArray[np.float64]  # left, top, width, height
    velocity: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    hits: int = 1
    misses: int = 0

    @property
    def center(self) -> npt.NDArray[np.float64]:
        return self.box[:2] + self.box[2:] / 2

    def predicted_center(self) -> npt.NDArray[np.float64]:
        return self.center + self.velocity * (1 + self.misses)


class CentroidTracker:
    """Hungarian assignment on distance between detections and each track's
    constant-velocity predicted centre, with a distance gate.

    Tracks are reported only once confirmed (min_hits consecutive-ish hits),
    the SORT convention: a one-frame blob shouldn't become an identity.
    """

    def __init__(self, gate_px: float = 25.0, max_misses: int = 5, min_hits: int = 3) -> None:
        self.gate_px, self.max_misses, self.min_hits = gate_px, max_misses, min_hits
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, dets: Detections) -> list[tuple[int, npt.NDArray[np.float64]]]:
        pred = np.array([t.predicted_center() for t in self._tracks]).reshape(-1, 2)
        centers = dets[:, :2] + dets[:, 2:] / 2
        matched_t: set[int] = set()
        matched_d: set[int] = set()
        if len(pred) and len(centers):
            dist = np.linalg.norm(pred[:, None] - centers[None], axis=2)
            for ti, di in zip(*linear_sum_assignment(dist), strict=True):
                if dist[ti, di] <= self.gate_px:
                    track = self._tracks[ti]
                    new_center = centers[di]
                    step = (new_center - track.center) / (1 + track.misses)
                    track.velocity = 0.5 * track.velocity + 0.5 * step  # light smoothing
                    track.box, track.hits, track.misses = dets[di].copy(), track.hits + 1, 0
                    matched_t.add(ti)
                    matched_d.add(di)
        for ti, track in enumerate(self._tracks):
            if ti not in matched_t:
                track.misses += 1
        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]
        for di in range(len(dets)):
            if di not in matched_d:
                self._tracks.append(_Track(self._next_id, dets[di].copy()))
                self._next_id += 1
        return [(t.track_id, t.box.copy()) for t in self._tracks if t.misses == 0 and t.hits >= self.min_hits]


class SurgeAlert:
    """Per-track speed over a short window -> raw trigger -> hysteresis."""

    def __init__(self, fps: float, config: AlertConfig, window_frames: int = 3) -> None:
        self.fps, self.config, self.window = fps, config, window_frames
        self._history: dict[int, list[tuple[int, npt.NDArray[np.float64]]]] = {}
        self._machines: dict[int, Hysteresis] = {}
        self._open: dict[int, float] = {}
        self.alarms: list[Alarm] = []

    def update(self, frame_idx: int, centers: dict[int, npt.NDArray[np.float64]]) -> None:
        now = frame_idx / self.fps
        for tid, c in centers.items():
            hist = self._history.setdefault(tid, [])
            hist.append((frame_idx, c))
            del hist[: -self.window - 1]
            raw = False
            if len(hist) > self.window:
                (f0, c0), (f1, c1) = hist[0], hist[-1]
                speed = float(np.linalg.norm(c1 - c0)) * self.fps / (f1 - f0)
                raw = speed > self.config.trigger_threshold
            machine = self._machines.setdefault(
                tid, Hysteresis(self.config.k_enter, self.config.n_window, self.config.m_exit)
            )
            was = machine.active
            if machine.update(raw) and not was:
                self._open[tid] = now
            elif was and not machine.active:
                self.alarms.append(Alarm("surge", self._open.pop(tid), now, tid))

    def finish(self, end_s: float) -> list[Alarm]:
        for tid, start in self._open.items():
            self.alarms.append(Alarm("surge", start, end_s, tid))
        self._open.clear()
        return self.alarms


@dataclass(frozen=True)
class PipelineResult:
    hyp_boxes: Boxes
    alarms: list[Alarm]
    frame_latency_ns: list[int]
    scene_changes: int


PIXEL_SURGE = AlertConfig(trigger_threshold=100.0, k_enter=4, n_window=6, m_exit=8)  # px/s


def run_pixel_pipeline(
    frames: npt.NDArray[np.uint8], fps: float, config: AlertConfig = PIXEL_SURGE
) -> PipelineResult:
    detector, tracker, alert = BlobDetector(), CentroidTracker(), SurgeAlert(fps, config)
    rows: list[list[float]] = []
    latency: list[int] = []
    for t, frame in enumerate(frames):
        t0 = time.perf_counter_ns()
        tracks = tracker.update(detector.detect(frame))
        alert.update(t, {tid: box[:2] + box[2:] / 2 for tid, box in tracks})
        latency.append(time.perf_counter_ns() - t0)
        rows.extend([t + 1, tid, *box.tolist(), 1.0] for tid, box in tracks)
    boxes = np.asarray(rows, dtype=np.float64).reshape(-1, 7)
    return PipelineResult(boxes, alert.finish(len(frames) / fps), latency, detector.scene_changes)


# 2.5 m/s, not 2.0: at 2.0 the suite measured ~1 false alarm per camera-day,
# every one a genuine stretch of ordinary walking above 2 m/s (the OU tail,
# P(speed > 2) = exp(-2^2 / (2 * 0.4^2)) ~ 4e-6 per sample) -- the threshold sat
# inside the normal speed distribution. At 2.5 that tail is ~3e-9. Choosing
# this separation from data is what Phase 4-5's conformal calibration does.
BEV_SURGE = AlertConfig(trigger_threshold=2.5, k_enter=3, n_window=4, m_exit=2)  # m/s


def surge_alarms_from_tracks(
    tracks: npt.NDArray[np.float64], fps: float, config: AlertConfig = BEV_SURGE, lag: int = 2
) -> list[Alarm]:
    """The same surge rule on ground-plane tracks (frame, id, x_m, y_m), for the
    multi-day kinematic suites: speed over `lag` samples, vectorised per track,
    then the batch hysteresis."""
    alarms: list[Alarm] = []
    for tid in np.unique(tracks[:, 1]):
        rows = tracks[tracks[:, 1] == tid]
        rows = rows[np.argsort(rows[:, 0], kind="stable")]
        speed = np.zeros(len(rows))
        speed[lag:] = np.linalg.norm(rows[lag:, 2:4] - rows[:-lag, 2:4], axis=1) * fps / lag
        raw = speed > config.trigger_threshold
        for start, end in hysteresis_intervals(raw, config.k_enter, config.n_window, config.m_exit):
            alarms.append(
                Alarm("surge", float(rows[start, 0] - 1) / fps, float(rows[end - 1, 0]) / fps, int(tid))
            )
    return alarms
