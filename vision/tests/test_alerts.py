"""The alert state machine: batch == streaming, and the single-frame-spike
guarantee -- with the control showing the test can see a spike at all."""

import numpy as np
import pytest
from pydantic import ValidationError

from vision.alerts import AlertConfig, Hysteresis, hysteresis_intervals


def _streaming_intervals(raw: np.ndarray, k: int, n: int, m: int) -> list[tuple[int, int]]:
    machine = Hysteresis(k, n, m)
    active = np.array([machine.update(bool(r)) for r in raw])
    edges = np.diff(np.concatenate([[0], active.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True))


@pytest.mark.parametrize("seed", range(20))
def test_batch_matches_streaming(seed: int) -> None:
    rng = np.random.default_rng(seed)
    k = int(rng.integers(1, 5))
    n, m = k + int(rng.integers(0, 4)), int(rng.integers(1, 6))
    # Sparse triggers with occasional bursts: exercises the batch version's
    # idle-skipping path as well as its normal stepping.
    raw = rng.random(2000) < rng.choice([0.02, 0.2, 0.6])
    raw[rng.integers(0, 1900) :][:30] = True
    assert hysteresis_intervals(raw, k, n, m) == _streaming_intervals(raw, k, n, m)


def test_single_frame_spikes_never_alert() -> None:
    raw = np.zeros(1000, dtype=bool)
    raw[::7] = True  # isolated one-frame spikes, as dense as k=2-of-n=3 allows without pairs
    cfg = AlertConfig(trigger_threshold=1.0, k_enter=2, n_window=3, m_exit=1)
    assert hysteresis_intervals(raw, cfg.k_enter, cfg.n_window, cfg.m_exit) == []


def test_negative_control_without_hysteresis_a_spike_alerts() -> None:
    raw = np.zeros(100, dtype=bool)
    raw[50] = True
    assert hysteresis_intervals(raw, 1, 1, 1) == [(50, 51)]


def test_alert_config_refuses_single_frame_entry() -> None:
    with pytest.raises(ValidationError):
        AlertConfig(trigger_threshold=1.0, k_enter=1, n_window=3, m_exit=1)
    with pytest.raises(ValidationError):
        AlertConfig(trigger_threshold=1.0, k_enter=4, n_window=3, m_exit=1)


def test_sustained_trigger_alerts_and_clears() -> None:
    raw = np.zeros(100, dtype=bool)
    raw[10:20] = True
    # Enters at the 3rd consecutive trigger (index 12); exits after 4 clear samples.
    assert hysteresis_intervals(raw, 3, 4, 4) == [(12, 23)]
