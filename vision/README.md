# Agrios vision — perception cell

Real-time CCTV perception for Agrios: RTSP ingest, detection, tracking, pose, behavioural analytics, and ground-plane projection into the C++ bridge. It is built in phases, and **Phase 1 is the evaluation harness**, built before any model. The rule is that no model goes in until there is something to measure it with, and nothing passes that measurement if a negative control would pass it too.

Status: **Phase 1 of 7 — evaluation harness and synthetic streams.** There is no model, RTSP or bridge code yet; see [Roadmap](#roadmap).

## What Phase 1 contains

| Module | What it is |
|---|---|
| [`eval/manifest.py`](eval/manifest.py) | Pydantic schema for what an evaluation runs. Synthetic sequences are `generator + seed + params`. Recorded footage is a path that **must be under `data/`** (gitignored), rejected at validation otherwise. Ground-truth boxes use the MOTChallenge CSV layout. |
| [`eval/metrics.py`](eval/metrics.py) | CLEAR-MOT (MOTA/MOTP), IDF1, event recall, False Alarms per Camera-Day (FACD) with an exact Poisson upper bound, GAI error, and latency percentiles with a bootstrap CI and a machine stamp. |
| [`eval/gate.py`](eval/gate.py) | The CI gate: runs a manifest, pools counts per suite, checks the manifest's criteria, writes `report.json` and `summary.md`, and exits nonzero on failure. `--control noise\|shuffle` corrupts the input while keeping the ground truth. |
| [`synth/`](synth/) | Seeded generators. **Scene**: pixel-level targets with an occluding pillar, sensor noise, an illumination drop, the IR night switch, camera shake and a velocity surge. **Kinematics**: camera-days of ground-plane tracks (Ornstein-Uhlenbeck motion, measurement jitter, single-sample outliers, Poisson-timed surges). **Gait**: quadruped footfall contacts with a known stance asymmetry. |
| [`alerts.py`](alerts.py) | The k-of-n enter / m-clear exit hysteresis that every alert passes through. `AlertConfig` enforces `k_enter >= 2`, so a single-frame spike can't alert. |
| [`reference/`](reference/) | A classical stand-in pipeline (MOG2, then blobs, then a constant-velocity tracker, then a surge alert). It gives the gate something to run end to end. **Its numbers are not model performance.** |
| [`privacy.py`](privacy.py) | Ensures no media file is tracked by git, and that `data/` is ignored. |

## Running it

```bash
uv sync --all-extras --all-groups
uv run pytest vision -q
uv run ruff check vision && uv run ruff format --check vision && uv run mypy
uv run python -m vision.eval.gate --manifest vision/eval/suites/phase1.json --out vision-results
uv run python -m vision.eval.gate --manifest vision/eval/suites/phase1.json --out vision-results --control noise    # must FAIL
uv run python -m vision.eval.gate --manifest vision/eval/suites/phase1.json --out vision-results --control shuffle  # must FAIL
```

## Measured results (reference pipeline, synthetic fixtures)

Same seeds on three platforms -- Windows 11, Linux x86-64 and Linux arm64 (both Debian containers; arm64 emulated under QEMU) -- and the metrics agree to the last printed digit. That includes the scene recall of exactly 3/4, the floor with no margin. Latency doesn't agree, and isn't expected to. Each control's corruption is seeded per sequence. CI repeats this on native arm64 (`ubuntu-24.04-arm`).

| Suite | Metric | Intact | Noise control | Shuffle control | Gate bound |
|---|---|---|---|---|---|
| scene (5 seq, 1,500 frames) | event recall | 0.75 (3/4) | 0 | 0.5 | ≥ 0.75 |
| | false alarms | 0 | 0 | 1 | ≤ 0 |
| | MOTA | 0.4963 | 0 | −0.2026 | ≥ 0.40 |
| | IDF1 | 0.4587 | 0 | 0.0043 | ≥ 0.35 |
| kinematics (3 camera-days) | event recall | 1.0 (33/33) | 0 | 0 | ≥ 0.95 |
| | false alarms | 0 | 988 | 1,053 | — |
| | FACD, one-sided 95% upper | 0.9986 | 347.1 | 369.3 | ≤ 1.0 |
| gait (3 seq) | max GAI error (points) | 1.16 | 23.6 | 4.95 | ≤ 2.0 |

Reference pipeline frame latency (n = 1,500 frames, 320×240):

| Machine | p50 ms [95% CI] | p95 | p99 | max |
|---|---|---|---|---|
| i7-1255U, Windows 11, AC, "Balanced" plan, run 1 | 3.029 [2.995–3.057] | 4.334 | 5.091 | 32.574 |
| same, run 2 | 3.796 [3.735–3.847] | 6.829 | 8.568 | 13.944 |
| same, run 3 | 3.566 [3.532–3.604] | 6.539 | 9.321 | 17.184 |
| same CPU, Docker linux/amd64 (WSL2) | 2.515 [2.461–2.572] | 4.168 | 5.831 | 11.106 |
| same CPU, Docker linux/arm64 under QEMU emulation | 172.031 [167.624–176.854] | 386.151 | 571.081 | 1019.491 |

Runs 1–3 differ by up to 25% at p50 on the same machine. The arm64 row is instruction-set emulation, about 50× slower: it shows the metrics are architecture-independent, and says nothing about a Pi 5. That is the laptop's power and thermal management, and it is why every latency figure here carries its machine stamp. It's also why no latency number is gated in CI.

**How to read these numbers:**
- **The bounds are regression floors, not targets.** They are set from the measured run minus a margin (the scene recall bound has none), so a change that makes things worse fails CI.
- **FACD ≤ 1.0 is the rule of three at work.** Zero false alarms in 3 camera-days gives a 95% upper bound of −ln(0.05)/3 ≈ 0.9986 per camera-day. Claiming "≤ 1 false alarm per camera-day" needs at least 3 alarm-free camera-days, on synthetic data here and, in Phase 7, on every real camera.
- **Synthetic FACD validates the harness arithmetic, not the cameras.** Real FACD comes only from shadow-mode runs on recorded footage.
- **The reference pipeline's MOTA ≈ 0.5 is its own limitation, diagnosed rather than tuned.** `report.json` splits each scene's misses into ground-truth boxes no hypothesis touches and boxes only partially covered (IoU between 0 and 0.5). The remainder are covered but matched to another target. On `scene-calm`, 188 of 275 misses are partial overlaps; on `scene-crowd-surge`, 400 of 564. Merged blobs (crossing targets) and MOG2's warm-up ghosts produce exactly this pattern. On `scene-ir-shake-surge` it flips: 603 of 692 misses have no overlap at all, because the scene-change guard drops whole frames. Phase 3's detector replaces this pipeline.
- **The one missed surge** is on the IR-switch-plus-shake fixture. The stand-in's scene-change guard suppresses 23 frames, and shake keeps MOG2 noisy. Phase 2's tamper and quality detectors handle that case properly.

## Design decisions

- **Two fixture levels, because of the rule of three.** Pixels can't be rendered for camera-days in CI, but false-alarm statistics need camera-days. Short pixel fixtures test the pixel path; long track-level fixtures (a camera-day in under a second, vectorised with `lfilter`) test the alert statistics.
- **MOTA and IDF1 are implemented here and cross-checked against py-motmetrics** to 1e-9, along with exact counts of misses, false positives and switches. Five seeded fixtures each exercise every error type; the test asserts that, so the agreement can't be vacuous. py-motmetrics 1.4.0 still calls `np.asfarray`, which NumPy 2.0 removed, so the test restores exactly that function (test-only). The comparison therefore runs motmetrics' own IoU code, not ours.
- **Suite metrics are pooled counts**, as in py-motmetrics' OVERALL row, not averages of per-sequence ratios.
- **The BEV surge threshold is 2.5 m/s, chosen from data.** At 2.0 m/s the kinematic suite measured about 1 false alarm per camera-day. Each one was a genuine stretch of ordinary walking: P(speed > 2) = exp(−2²/(2·0.4²)) ≈ 4×10⁻⁶ per sample, so 2.0 sat inside the normal speed distribution. At 2.5 m/s that tail is about 3×10⁻⁹. Phases 4–5 make this kind of choice with conformal calibration instead of by hand.
- **The negative controls found a real bug.** The noise control produces zero tracker output, which crashed `_by_frame` (`np.split` of an empty array yields one chunk, not zero). It's fixed, and the empty-hypothesis case is now a unit test.
- **A gate with nothing to check fails.** A suite without criteria, criteria for a suite that never ran, or a run with zero checks all fail, because an empty `all()` is True. `--expect-fail`, which CI uses for the controls, succeeds only if *every* suite fails a check, so one suite catching the corruption can't cover for another.
- **The kinematic suites match events by track.** Alarm ids there are ground-truth ids, so an alarm on track A during track B's surge counts as a false alarm instead of crediting B.
- **Box arrays are validated.** A duplicated id within one frame is rejected, both in the metrics and in `read_mot_csv`, because it would push IDF1 above 1.
- **No property-based testing library.** Hypothesis is MPL-2.0, outside the MIT/BSD/Apache rule, so the tests use seeded parametrisation instead.
- **Lint and types from day one.** ruff (a broad rule set) and `mypy --strict` gate `vision/` in CI. Agent code that predates them isn't held to them yet.

## Known gaps in the controls

These came out of the fresh-context review and aren't closed yet:
- **The scene `false_alarms` bound has no working noise control.** Pure noise trips the scene-change guard on every frame, so nothing alarms and the bound passes. The other scene bounds catch the noise control. The bound is still real under shuffle (1 false alarm).
- **Shuffling a gait sequence keeps each limb's contact fraction**, and the GAI of those fractions equals the true GAI. The shuffle control therefore fails only because it breaks stance *run lengths*. An estimator that used contact fractions instead of runs would pass it. Phase 4 adds a control that breaks the marginals too, such as swapping limbs within contralateral pairs.
- `SurgeAlert` keeps per-track state for tracks that have ended. That's harmless on 20 s fixtures; Phase 2's long-running ingest needs it pruned.

## Licences

Runtime dependencies (`vision` extra) are numpy and scipy (BSD-3) and opencv-python-headless (Apache-2.0; bundles FFmpeg under LGPL-2.1, accepted as a dynamically linked runtime dependency). Dev-only dependencies are motmetrics (MIT, with pandas BSD-3 and xmltodict MIT), ruff (MIT) and mypy (MIT). The policy for model weights, in force from Phase 3:
- a CI gate on declared licences (MIT/BSD/Apache, plus the FFmpeg LGPL exception);
- a model manifest recording each weight file's training datasets;
- self-trained re-ID and action models;
- weights downloaded by script, never committed.

## What it's for

The perception cell watches livestock: cattle and alpacas.

| Goal | What the system does |
|---|---|
| Health | Gait and lameness per animal (stance asymmetry from footfalls), activity levels, and an animal standing apart from the herd or unusually still |
| Behaviour | Aggression within the herd, by time of day and by pair |
| Predators | Large animals at the perimeter, and at the alpaca house above all: a puma there is the highest-stakes alarm in the project, so it favours recall over false alarms |
| Camera health | A camera that's covered, moved, blurred or frozen raises its own alert, so the system never goes silently blind |

The gait pipeline is developed on the synthetic quadruped fixtures here and, later, on licence-checked animal footage. Free tools only, including for alerts.

## Roadmap

The user interface is a separate track, the Agrios Console ([docs/console-plan.md](../docs/console-plan.md)). Its milestones C2–C4 land alongside the phases below.

| Phase | Scope | Notes carried forward |
|---|---|---|
| 1 ✅ | Harness, synthetic streams, negative controls | — |
| 2 | RTSP ingest (drop-on-full, `CAP_PROP_BUFFERSIZE=1`), tamper/quality, EXIF scrub | CI streams synthetic video through a local RTSP server, so no camera is needed. A camera that can't see properly (tamper, blur, IR switch, shake) raises its own alert, never silence: the Phase 1 scene-change guard dropped 23 frames and missed a real surge. A decoy control (normal-looking frames with objects absent from the ground truth) closes the scene false-alarm gap below. |
| 3 | ONNX detector (D-FINE / RT-DETR), ByteTrack + OSNet, BEV homography | Add HOTA (the MOTChallenge headline metric) via TrackEval; benchmark DirectML vs OpenVINO on the iGPU. Console C2: setup wizard (calibration, zones) and Cameras screen. |
| 4 | Animal pose + gait (mmaction2 GCNs, not Shift-GCN: CC BY-NC) | Needs a side-on view at body height (camera question Q2). Public animal datasets are used to test, not to train, unless their licence allows it. |
| 5 | Herd behaviour and predator alerts + conformal calibration | Calibrate on block maxima: video windows aren't exchangeable. A predator alarm favours recall, and each alarm carries a photo to check, against alarm fatigue. A miss-rate claim needs about 3/x staged events with no misses (rule of three). |
| 6 | Bridge, `/agrios_alert_telemetry`, Mega fail-safe on the Pi 5 | Pi-side ingest re-stamps poses with the Pi's clock: Windows and Pi monotonic clocks aren't comparable. Console C3: incident review and analytics. |
| 7 | Shadow mode on recorded footage, end-to-end latency, final audit | ≥ 3 alarm-free camera-days per camera before any FACD claim. Console C4: evidence-gated arming and usability test. |
