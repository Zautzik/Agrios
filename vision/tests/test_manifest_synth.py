"""Manifest validation and generator determinism."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from vision.eval.manifest import (
    FileSource,
    GaitParams,
    GroundTruthEvent,
    KinematicsParams,
    Manifest,
    SceneParams,
    read_mot_csv,
    write_mot_csv,
)
from vision.synth.gait import generate_gait
from vision.synth.kinematics import generate_kinematics
from vision.synth.scene import generate_scene

PHASE1 = Path(__file__).parents[1] / "eval" / "suites" / "phase1.json"


def test_phase1_manifest_validates() -> None:
    m = Manifest.load(PHASE1)
    assert {s.source.kind for s in m.sequences} == {"scene", "kinematics", "gait"}


@pytest.mark.parametrize("path", ["data/cam1/2026-11-03.mp4", "data/x.mkv"])
def test_file_source_under_data_is_accepted(path: str) -> None:
    FileSource.model_validate({"kind": "file", "path": path})


@pytest.mark.parametrize(
    "path", ["footage/cam1.mp4", "/abs/data/x.mp4", "data/../x.mp4", "vision/data/x.mp4"]
)
def test_file_source_outside_data_is_rejected(path: str) -> None:
    with pytest.raises(ValidationError, match="under data/"):
        FileSource.model_validate({"kind": "file", "path": path})


def test_manifest_rejects_duplicate_ids_and_backwards_events() -> None:
    seq = {"id": "a", "source": {"kind": "scene", "seed": 1}}
    with pytest.raises(ValidationError, match="unique"):
        Manifest.model_validate({"sequences": [seq, seq]})
    with pytest.raises(ValidationError, match="before it starts"):
        GroundTruthEvent(kind="surge", start_s=5, end_s=4)
    with pytest.raises(ValidationError):
        Manifest.model_validate({"sequences": [seq], "typo_field": 1})


def test_mot_csv_round_trip(tmp_path: Path) -> None:
    gt = generate_scene(SceneParams(duration_s=2), seed=1).gt_boxes
    write_mot_csv(gt, tmp_path / "gt.txt")
    assert np.allclose(read_mot_csv(tmp_path / "gt.txt"), gt, atol=1e-3)


@pytest.mark.parametrize(
    ("generate", "params", "field"),
    [
        (generate_scene, SceneParams(duration_s=3, shake_px=1.0, surge_s=1.0), "frames"),
        (generate_kinematics, KinematicsParams(duration_s=3600), "tracks"),
        (generate_gait, GaitParams(asymmetry=0.2), "contacts"),
    ],
)
def test_generators_are_deterministic_per_seed(
    generate: Callable[[Any, int], object], params: object, field: str
) -> None:
    a, b, other = (getattr(generate(params, s), field) for s in (5, 5, 6))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, other)  # control: the seed actually matters


def test_scene_ground_truth_and_events() -> None:
    p = SceneParams(duration_s=4, n_targets=3, surge_s=2.0)
    s = generate_scene(p, seed=3)
    assert s.frames.shape == (60, 240, 320, 3)
    assert len(s.gt_boxes) == 60 * 3
    assert s.events[0].start_s == 2.0
    assert s.events[0].track_ids == (1,)


def test_kinematics_stays_in_arena_and_injects_surges() -> None:
    # surge rate 4/h: ~24 expected, so "no surges at all" isn't a coin flip of the seed.
    p = KinematicsParams(
        duration_s=6 * 3600, position_noise_m=0, spike_rate_per_track_hour=0, surge_rate_per_hour=4
    )
    s = generate_kinematics(p, seed=4)
    assert s.tracks[:, 2].min() >= 0
    assert s.tracks[:, 2].max() <= p.arena_m[0]
    assert s.tracks[:, 3].min() >= 0
    assert s.tracks[:, 3].max() <= p.arena_m[1]
    assert len(s.events) > 0


def test_gait_true_gai_follows_asymmetry() -> None:
    assert generate_gait(GaitParams(asymmetry=0.0), seed=1).true_gai == 0.0
    lame = generate_gait(GaitParams(asymmetry=0.25, lame_limb="RH"), seed=1)
    assert lame.pair == ("LH", "RH")
    assert lame.true_gai == pytest.approx(100 * 0.25 / 0.875)
