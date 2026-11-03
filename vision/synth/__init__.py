"""Seeded procedural streams. Every generator is a pure function of
(params, seed): CI regenerates fixtures instead of committing video, and no
real footage is ever needed to test the pipeline end to end."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from vision.eval.manifest import Boxes, GroundTruthEvent


@dataclass(frozen=True)
class SceneSequence:
    frames: npt.NDArray[np.uint8]  # (T, H, W, 3) BGR, OpenCV's channel order
    gt_boxes: Boxes
    events: tuple[GroundTruthEvent, ...]
    fps: float

    @property
    def duration_s(self) -> float:
        return len(self.frames) / self.fps


@dataclass(frozen=True)
class KinematicSequence:
    # (N, 4): frame (1-based), track id, x_m, y_m -- *measured* positions,
    # i.e. with the noise and outliers a real tracker's output would carry.
    tracks: npt.NDArray[np.float64]
    events: tuple[GroundTruthEvent, ...]
    fps: float
    duration_s: float


@dataclass(frozen=True)
class GaitSequence:
    contacts: npt.NDArray[np.bool_]  # (T, 4) in LIMBS order
    fps: float
    true_stance_s: dict[str, float] = field(default_factory=dict)  # mean configured stance per limb
    true_gai: float = 0.0  # GAI of the lame limb's contralateral pair
    pair: tuple[str, str] = ("LF", "RF")


LIMBS = ("LF", "RF", "LH", "RH")
