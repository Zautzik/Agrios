"""Track-level synthetic streams in ground-plane metres, long enough to measure
false alarms per camera-day.

Velocities follow an Ornstein-Uhlenbeck process (mean-reverting random
motion: animals and people wander, they don't drift off at constant
velocity), integrated to positions that reflect off the arena walls. On top
of the true motion, measured positions get Gaussian jitter and rare
single-sample outliers -- the detector glitches and mis-associations the
alert hysteresis has to ignore -- and Poisson-timed surges, where one track
runs at a fixed high speed, are the injected events.

Everything is vectorised over time (scipy.signal.lfilter for the OU recursion,
a fold for wall reflection), so a simulated camera-day takes well under a
second instead of a per-sample Python loop.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter

from vision.eval.manifest import GroundTruthEvent, KinematicsParams
from vision.synth import KinematicSequence


def _reflect(x: npt.NDArray[np.float64], length: float) -> npt.NDArray[np.float64]:
    """Folds an unbounded coordinate into [0, length], as a wall bounce would."""
    period = np.mod(x, 2 * length)
    out: npt.NDArray[np.float64] = length - np.abs(period - length)
    return out


def generate_kinematics(p: KinematicsParams, seed: int) -> KinematicSequence:
    rng = np.random.default_rng(seed)
    n = round(p.duration_s * p.fps)
    dt = 1.0 / p.fps
    a = np.exp(-dt / p.velocity_tau_s)
    # x[t] = a x[t-1] + sqrt(1 - a^2) sigma w[t] has stationary std sigma.
    shocks = rng.normal(0, p.speed_sigma_mps * np.sqrt(1 - a * a), (p.n_tracks, n, 2))
    vel = lfilter([1.0], [1.0, -a], shocks, axis=1)

    events: list[GroundTruthEvent] = []
    n_surges = rng.poisson(p.surge_rate_per_hour * p.duration_s / 3600)
    surge_len = max(1, round(p.surge_duration_s * p.fps))
    for start in np.sort(rng.integers(0, max(1, n - surge_len), n_surges)):
        track = int(rng.integers(p.n_tracks))
        heading = rng.uniform(0, 2 * np.pi)
        vel[track, start : start + surge_len] = p.surge_speed_mps * np.array(
            [np.cos(heading), np.sin(heading)]
        )
        events.append(
            GroundTruthEvent(
                kind="surge",
                start_s=start * dt,
                end_s=(start + surge_len) * dt,
                track_ids=(track + 1,),
            )
        )

    start_pos = rng.uniform([0, 0], p.arena_m, (p.n_tracks, 1, 2))
    true_pos = start_pos + np.cumsum(vel * dt, axis=1)
    for axis, length in enumerate(p.arena_m):
        true_pos[..., axis] = _reflect(true_pos[..., axis], length)

    measured = true_pos + rng.normal(0, p.position_noise_m, true_pos.shape)
    spike_prob = p.spike_rate_per_track_hour * dt / 3600
    spikes = rng.random((p.n_tracks, n)) < spike_prob
    spike_heading = rng.uniform(0, 2 * np.pi, (p.n_tracks, n))
    measured += (spikes * p.spike_magnitude_m)[..., None] * np.stack(
        [np.cos(spike_heading), np.sin(spike_heading)], axis=-1
    )

    frame = np.tile(np.arange(1, n + 1, dtype=np.float64), p.n_tracks)
    track_id = np.repeat(np.arange(1, p.n_tracks + 1, dtype=np.float64), n)
    tracks = np.column_stack([frame, track_id, measured.reshape(-1, 2)])
    return KinematicSequence(tracks, tuple(events), p.fps, n * dt)
