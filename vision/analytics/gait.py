"""Gait timing primitives: stance durations from footfall contacts, and the
Gait Asymmetry Index.

GAI here is Robinson's symmetry index (Robinson, Herzog & Nigg 1987),
100 * |L - R| / (0.5 * (L + R)), on mean stance duration of a contralateral
limb pair. 0 is perfect symmetry; it's unitless, so it compares across
animals of different size and cadence. Phase 4 feeds it contact timings
estimated from RTMPose keypoints; Phase 1 feeds it the synthetic generator's
contact sequences, so the metric and its error are tested before any model is.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def gait_asymmetry_index(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        raise ValueError(f"stance durations must be positive, got {left}, {right}")
    return 100.0 * abs(left - right) / (0.5 * (left + right))


def stance_durations(contacts: npt.NDArray[np.bool_], fps: float) -> list[float]:
    """Durations (s) of each complete stance -- each maximal run of contact
    frames. Runs touching the first or last frame are dropped: their true
    length is unknown, and counting them would bias the mean short."""
    c = np.asarray(contacts, dtype=np.int8)
    edges = np.diff(np.concatenate([[0], c, [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    keep = (starts > 0) & (ends < len(c))
    return [float(n) / fps for n in (ends - starts)[keep]]
