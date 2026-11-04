"""Temporal alert state machine shared by every behavioural detector.

No per-frame score ever raises an alert directly. A raw per-frame trigger
(speed over threshold, polygon entry, asymmetry over bound, ...) only feeds a
k-of-n entry / m-consecutive-clear exit hysteresis, so a single-frame spike --
a detector glitch, a mis-association, one corrupted decode -- can never alert
by construction once k_enter >= 2. AlertConfig enforces that floor; Hysteresis
itself accepts k_enter = 1 only so the test suite can show the spike *would*
alert without it (the negative control for that guarantee).

The thresholds here are hand-set in Phase 1. Phases 4-5 calibrate them with
split conformal prediction (vision/conformal.py) against a false-alarm budget.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Hysteresis:
    """Streaming k-of-n enter / m-consecutive-clear exit state machine.

    Enters the active state once at least k_enter of the last n_window raw
    triggers were true; leaves it after m_exit consecutive false triggers.
    """

    def __init__(self, k_enter: int, n_window: int, m_exit: int) -> None:
        if not 1 <= k_enter <= n_window:
            raise ValueError(f"need 1 <= k_enter <= n_window, got {k_enter}, {n_window}")
        if m_exit < 1:
            raise ValueError(f"m_exit must be >= 1, got {m_exit}")
        self.k_enter, self.n_window, self.m_exit = k_enter, n_window, m_exit
        self._window: deque[bool] = deque(maxlen=n_window)
        self._hits = 0  # true count inside _window, kept so update() is O(1)
        self._clear_run = 0
        self.active = False

    def update(self, raw: bool) -> bool:
        if len(self._window) == self.n_window and self._window[0]:
            self._hits -= 1
        self._window.append(raw)
        self._hits += raw
        if self.active:
            self._clear_run = 0 if raw else self._clear_run + 1
            if self._clear_run >= self.m_exit:
                self.active = False
        elif self._hits >= self.k_enter:
            self.active, self._clear_run = True, 0
        return self.active

    @property
    def idle(self) -> bool:
        """Inactive with no true trigger in the window: the state can't change
        until the next true trigger arrives."""
        return not self.active and self._hits == 0


def hysteresis_intervals(
    raw: npt.NDArray[np.bool_], k_enter: int, n_window: int, m_exit: int
) -> list[tuple[int, int]]:
    """Batch form of Hysteresis over a whole sequence: the [start, end) runs of
    samples for which update() returned True.

    Same semantics as feeding raw through Hysteresis.update() one sample at a
    time (the test suite checks that equivalence on random sequences), but it
    skips the stretches where the machine is idle with no trigger in its
    window. That matters for the multi-camera-day kinematic suites, where
    triggers are rare and a per-sample Python loop would dominate runtime.
    """
    trues = np.flatnonzero(raw)
    intervals: list[tuple[int, int]] = []
    machine = Hysteresis(k_enter, n_window, m_exit)
    i, n = 0, len(raw)
    next_true = 0  # index into trues
    start = -1
    while i < n:
        if machine.idle:
            # Idle and nothing in the window: nothing can happen until the next
            # true trigger, and an all-false window counts the same as an empty
            # one, so jump straight there with a fresh window.
            while next_true < len(trues) and trues[next_true] < i:
                next_true += 1
            if next_true == len(trues):
                break
            i = int(trues[next_true])
            machine = Hysteresis(k_enter, n_window, m_exit)
        was_active = machine.active
        if machine.update(bool(raw[i])) and not was_active:
            start = i
        elif was_active and not machine.active:
            intervals.append((start, i))
        i += 1
    if machine.active:
        intervals.append((start, n))
    return intervals


class AlertConfig(BaseModel):
    """Thresholds for one raw trigger plus its hysteresis. Units of
    trigger_threshold belong to the detector (px/s in image space, m/s in
    BEV space)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trigger_threshold: float = Field(gt=0)
    # k_enter >= 2 is the "no alert from a single-frame spike" guarantee.
    k_enter: int = Field(ge=2)
    n_window: int = Field(ge=2)
    m_exit: int = Field(ge=1)

    @model_validator(mode="after")
    def _k_fits_window(self) -> AlertConfig:
        if self.k_enter > self.n_window:
            raise ValueError("k_enter must not exceed n_window")
        return self


@dataclass(frozen=True, slots=True)
class Alarm:
    """One alert interval raised by a detector, in sequence time."""

    kind: str
    start_s: float
    end_s: float
    track_id: int
