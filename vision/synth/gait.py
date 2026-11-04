"""Synthetic quadruped footfall contacts with a known stance asymmetry.

A lateral-sequence walk -- the footfall order cattle walk with: LH, LF, RH,
RF, a quarter-stride apart. Each limb is in stance for duty_factor of the
stride; the lame limb's stance is shortened by `asymmetry` (an animal
unloads a painful limb quickly). Stride period jitters per stride. The true
GAI is computed from the configured mean stances, so an estimator's error
against it includes the frame quantisation a real camera imposes.
"""

from __future__ import annotations

import numpy as np

from vision.analytics.gait import gait_asymmetry_index
from vision.eval.manifest import GaitParams
from vision.synth import LIMBS, GaitSequence

_PHASE = {"LH": 0.0, "LF": 0.25, "RH": 0.5, "RF": 0.75}
_CONTRALATERAL = {"LF": "RF", "RF": "LF", "LH": "RH", "RH": "LH"}


def generate_gait(p: GaitParams, seed: int) -> GaitSequence:
    rng = np.random.default_rng(seed)
    n = round(p.duration_s * p.fps)
    t = np.arange(n) / p.fps
    stance_frac = {limb: p.duty_factor * (1 - p.asymmetry if limb == p.lame_limb else 1.0) for limb in LIMBS}

    contacts = np.zeros((n, len(LIMBS)), dtype=bool)
    stride_start = 0.0
    while stride_start < p.duration_s:
        period = p.stride_s + rng.normal(0, p.stride_jitter_s)
        for k, limb in enumerate(LIMBS):
            on = stride_start + _PHASE[limb] * period
            contacts[(t >= on) & (t < on + stance_frac[limb] * period), k] = True
        stride_start += period

    true_stance = {limb: stance_frac[limb] * p.stride_s for limb in LIMBS}
    pair = tuple(sorted((p.lame_limb, _CONTRALATERAL[p.lame_limb]), key=LIMBS.index))
    left, right = pair
    return GaitSequence(
        contacts,
        p.fps,
        true_stance,
        gait_asymmetry_index(true_stance[left], true_stance[right]),
        (left, right),
    )
