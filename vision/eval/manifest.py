"""Validation manifest: which sequences an evaluation runs, where their ground
truth comes from, and the pass/fail criteria the gate holds them to.

One schema for both kinds of input. Synthetic sequences are described by
generator + seed + parameters, so CI regenerates them byte-identically instead
of committing video. Recorded footage is described by a path that must live
under data/ -- the gitignored directory real CCTV is confined to -- and is
rejected at validation time otherwise, so a manifest can't quietly point the
harness at footage somewhere it might get committed.

Ground-truth boxes use the MOTChallenge CSV layout (frame, id, left, top,
width, height, conf, class, visibility; frames 1-based), so recorded ground
truth can come straight out of standard annotation tools and the same files
are readable by py-motmetrics and TrackEval.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EventKind = Literal["surge", "intrusion", "lameness", "idle"]

# Column order of a box array everywhere in vision.eval: MOTChallenge's first
# six columns plus visibility. Kept as named indices rather than a structured
# dtype so arrays stay plain float64 and slice/compute cheaply.
FRAME, TRACK_ID, LEFT, TOP, WIDTH, HEIGHT, VISIBILITY = range(7)
BOX_COLUMNS = 7

Boxes = npt.NDArray[np.float64]  # (N, BOX_COLUMNS)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GroundTruthEvent(_Frozen):
    kind: EventKind
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    track_ids: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _ordered(self) -> GroundTruthEvent:
        if self.end_s < self.start_s:
            raise ValueError(f"event ends ({self.end_s}) before it starts ({self.start_s})")
        return self


# ---- Synthetic generator parameters ----


class SceneParams(_Frozen):
    """Pixel-level scene: coloured targets over a textured background."""

    width: int = Field(default=320, ge=64)
    height: int = Field(default=240, ge=64)
    fps: float = Field(default=15.0, gt=0)
    duration_s: float = Field(default=20.0, gt=0)
    n_targets: int = Field(default=3, ge=1, le=16)
    occluder: bool = True  # a full-height pillar targets pass behind
    illumination_drop_s: float | None = None  # global gain step to 45%
    ir_switch_s: float | None = None  # day -> night IR: grayscale, more gain/noise
    shake_px: float = Field(default=0.0, ge=0)  # per-frame camera translation jitter
    sensor_noise: float = Field(default=4.0, ge=0)
    surge_s: float | None = None  # target 1 accelerates at this time
    surge_duration_s: float = Field(default=1.5, gt=0)
    surge_factor: float = Field(default=4.0, gt=1)


class KinematicsParams(_Frozen):
    """Track-level stream in ground-plane metres, no pixels: cheap enough to
    simulate camera-days, which is what false-alarm statistics need."""

    fps: float = Field(default=2.0, gt=0)
    duration_s: float = Field(default=86_400.0, gt=0)
    n_tracks: int = Field(default=8, ge=1)
    arena_m: tuple[float, float] = (40.0, 30.0)
    speed_sigma_mps: float = Field(default=0.4, gt=0)  # stationary std of each velocity axis
    velocity_tau_s: float = Field(default=10.0, gt=0)  # Ornstein-Uhlenbeck time constant
    position_noise_m: float = Field(default=0.05, ge=0)
    spike_rate_per_track_hour: float = Field(default=2.0, ge=0)  # single-sample outliers
    spike_magnitude_m: float = Field(default=1.5, ge=0)
    surge_rate_per_hour: float = Field(default=0.5, ge=0)
    surge_speed_mps: float = Field(default=4.0, gt=0)
    surge_duration_s: float = Field(default=3.0, gt=0)


class GaitParams(_Frozen):
    """Quadruped footfall contacts (lateral-sequence walk) with an injected
    stance-duration asymmetry on one limb."""

    fps: float = Field(default=30.0, gt=0)
    duration_s: float = Field(default=30.0, gt=0)
    stride_s: float = Field(default=1.2, gt=0)
    duty_factor: float = Field(default=0.6, gt=0, lt=1)
    asymmetry: float = Field(default=0.0, ge=0, lt=0.5)  # fractional stance reduction, lame limb
    lame_limb: Literal["LF", "RF", "LH", "RH"] = "LF"
    stride_jitter_s: float = Field(default=0.01, ge=0)


# ---- Sources ----


class SceneSource(_Frozen):
    kind: Literal["scene"]
    seed: int
    params: SceneParams = SceneParams()


class KinematicsSource(_Frozen):
    kind: Literal["kinematics"]
    seed: int
    params: KinematicsParams = KinematicsParams()


class GaitSource(_Frozen):
    kind: Literal["gait"]
    seed: int
    params: GaitParams = GaitParams()


class FileSource(_Frozen):
    """Recorded footage plus its annotations. Read by vision.sources (Phase 2);
    Phase 1 only validates it."""

    kind: Literal["file"]
    path: PurePosixPath
    gt_boxes: PurePosixPath | None = None
    events: tuple[GroundTruthEvent, ...] = ()

    @field_validator("path", "gt_boxes")
    @classmethod
    def _under_data(cls, p: PurePosixPath | None) -> PurePosixPath | None:
        if p is None:
            return p
        if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != "data":
            raise ValueError(
                f"{p}: recorded footage and its annotations must live under data/ "
                "(gitignored), as a relative path without '..'"
            )
        return p


Source = Annotated[SceneSource | KinematicsSource | GaitSource | FileSource, Field(discriminator="kind")]


# ---- Criteria and the manifest itself ----


class GateCriteria(_Frozen):
    """Pass/fail bounds for one suite (all sequences sharing a source kind).
    Unset bounds aren't checked. These are regression floors set from measured
    runs, not performance claims -- see the gate report for the numbers."""

    min_event_recall: float | None = Field(default=None, ge=0, le=1)
    max_false_alarms: int | None = Field(default=None, ge=0)
    max_facd_upper95: float | None = Field(default=None, ge=0)
    min_mota: float | None = Field(default=None, le=1)
    min_idf1: float | None = Field(default=None, ge=0, le=1)
    max_gai_error: float | None = Field(default=None, ge=0)


class SequenceSpec(_Frozen):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    camera_id: str = Field(default="synthetic", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    source: Source


class Manifest(_Frozen):
    schema_version: Literal[1] = 1
    sequences: tuple[SequenceSpec, ...] = Field(min_length=1)
    criteria: dict[Literal["scene", "kinematics", "gait", "file"], GateCriteria] = {}

    @model_validator(mode="after")
    def _unique_ids(self) -> Manifest:
        ids = [s.id for s in self.sequences]
        if len(ids) != len(set(ids)):
            raise ValueError("sequence ids must be unique")
        return self

    @classmethod
    def load(cls, path: Path) -> Manifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


# ---- MOTChallenge CSV ----


def write_mot_csv(boxes: Boxes, path: Path) -> None:
    """Writes MOTChallenge ground-truth rows: conf = 1 and class = 1 for every
    row, since the harness has no per-row class yet."""
    out = np.column_stack(
        [boxes[:, :VISIBILITY], np.ones(len(boxes)), np.ones(len(boxes)), boxes[:, VISIBILITY]]
    )
    np.savetxt(path, out, delimiter=",", fmt=["%d", "%d", "%.3f", "%.3f", "%.3f", "%.3f", "%d", "%d", "%.4f"])


def read_mot_csv(path: Path) -> Boxes:
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    if raw.shape[1] < 6:
        raise ValueError(f"{path}: need at least 6 MOTChallenge columns, got {raw.shape[1]}")
    keys = raw[:, :2].astype(np.int64)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError(f"{path}: an id appears more than once in the same frame")
    vis = raw[:, 8] if raw.shape[1] >= 9 else np.ones(len(raw))
    return np.column_stack([raw[:, :6], vis]).astype(np.float64)
