"""Pixel-level synthetic scenes: coloured box targets moving over a textured
background, with the nuisances real CCTV has -- a pillar to pass behind,
sensor noise, a sudden illumination drop, the day-to-night IR switch, and
camera shake -- plus an injected velocity surge as the anomaly to detect.

Ground truth is exact by construction: every box is where the generator drew
it, full extent even when the pillar hides part of it (MOTChallenge counts
occluded objects too, with a visibility value -- recorded here per box).
Camera shake moves the whole image, so it moves the ground-truth boxes too.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from vision.eval.manifest import GroundTruthEvent, SceneParams
from vision.synth import SceneSequence

_PILLAR_WIDTH = 24
_PILLAR_GRAY = 60


def _background(p: SceneParams, rng: np.random.Generator) -> npt.NDArray[np.float32]:
    # Low-frequency texture (coarse noise upsampled), so the background isn't
    # flat -- a flat background makes background subtraction trivially easy.
    coarse = rng.normal(0, 1, (p.height // 16 + 1, p.width // 16 + 1, 3)).astype(np.float32)
    tex = cv2.resize(coarse, (p.width, p.height), interpolation=cv2.INTER_CUBIC)
    out: npt.NDArray[np.float32] = np.clip(np.array([95.0, 105.0, 100.0]) + 22.0 * tex, 0, 255).astype(
        np.float32
    )
    return out


def generate_scene(p: SceneParams, seed: int) -> SceneSequence:
    rng = np.random.default_rng(seed)
    n_frames = round(p.duration_s * p.fps)
    dt = 1.0 / p.fps
    bg = _background(p, rng)

    sizes = rng.uniform(18, 30, (p.n_targets, 2))
    pos = rng.uniform([0, 0], [p.width, p.height], (p.n_targets, 2)) - sizes / 2
    pos = np.clip(pos, 0, [p.width, p.height] - sizes)
    angle = rng.uniform(0, 2 * np.pi, p.n_targets)
    vel = rng.uniform(30, 50, p.n_targets)[:, None] * np.column_stack([np.cos(angle), np.sin(angle)])
    # Saturated, well-separated hues: targets differ from the background and
    # from each other, as people/animals in distinct clothing/coats would.
    hues = (np.arange(p.n_targets) * 180 // p.n_targets).astype(np.uint8)
    hsv = np.stack([hues, np.full_like(hues, 220), np.full_like(hues, 230)], axis=1)[None]
    colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0].astype(np.float32)

    pillar_x0 = p.width // 2 - _PILLAR_WIDTH // 2
    frames = np.empty((n_frames, p.height, p.width, 3), dtype=np.uint8)
    rows: list[list[float]] = []
    for t in range(n_frames):
        now = t * dt
        factor = np.ones(p.n_targets)
        if p.surge_s is not None and p.surge_s <= now < p.surge_s + p.surge_duration_s:
            factor[0] = p.surge_factor
        if t:
            pos += vel * factor[:, None] * dt
            for axis, limit in ((0, p.width), (1, p.height)):  # bounce off the frame edges
                lo, hi = pos[:, axis] < 0, pos[:, axis] + sizes[:, axis] > limit
                vel[lo | hi, axis] *= -1
                pos[:, axis] = np.clip(pos[:, axis], 0, limit - sizes[:, axis])

        img = bg.copy()
        for k in range(p.n_targets):
            x0, y0 = pos[k]
            x1, y1 = x0 + sizes[k, 0], y0 + sizes[k, 1]
            img[int(y0) : int(y1), int(x0) : int(x1)] = colors[k]
            hidden = max(0.0, min(x1, pillar_x0 + _PILLAR_WIDTH) - max(x0, pillar_x0))
            vis = 1.0 - hidden / sizes[k, 0] if p.occluder else 1.0
            rows.append([t + 1, k + 1, x0, y0, sizes[k, 0], sizes[k, 1], vis])
        if p.occluder:
            img[:, pillar_x0 : pillar_x0 + _PILLAR_WIDTH] = _PILLAR_GRAY

        noise = p.sensor_noise
        if p.illumination_drop_s is not None and now >= p.illumination_drop_s:
            img *= 0.45
        if p.ir_switch_s is not None and now >= p.ir_switch_s:
            # IR night mode: monochrome, auto-gain lifts the level and the noise with it.
            gray = img.mean(axis=2, keepdims=True) * 1.3
            img = np.repeat(gray, 3, axis=2)
            noise *= 2.0
        img += rng.normal(0, noise, img.shape).astype(np.float32)

        if p.shake_px > 0:
            dx, dy = rng.uniform(-p.shake_px, p.shake_px, 2)
            m = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
            img = cv2.warpAffine(img, m, (p.width, p.height), borderMode=cv2.BORDER_REFLECT).astype(
                np.float32, copy=False
            )
            for r in rows[-p.n_targets :]:
                r[2] += dx
                r[3] += dy
        frames[t] = np.clip(img, 0, 255).astype(np.uint8)

    events: tuple[GroundTruthEvent, ...] = ()
    if p.surge_s is not None:
        events = (
            GroundTruthEvent(
                kind="surge", start_s=p.surge_s, end_s=p.surge_s + p.surge_duration_s, track_ids=(1,)
            ),
        )
    return SceneSequence(frames, np.asarray(rows, dtype=np.float64), events, p.fps)
