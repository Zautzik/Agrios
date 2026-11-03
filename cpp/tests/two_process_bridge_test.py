"""Two-process bridge test: a real motor_control_rt_loop process on one side,
this separate Python process on the other, talking only through the POSIX
shared-memory channels.

Run it while motor_control_rt_loop is running with --report (CI starts the
loop first, then this script, then waits for the loop to exit):

    python cpp/tests/two_process_bridge_test.py --report /tmp/rt_report.json

What it checks:
  1. The agent's own tools (spatial_tools.read_joint_states /
     submit_spatial_intent) reach the live loop -- the exact path the agent
     uses, not a parallel test-only path.
  2. Every joint-state sample read across the process boundary is internally
     consistent. The loop publishes a_j = 0.3*sin(phase + 0.5*j), so any one
     sample must satisfy a_{j-1} + a_{j+1} = 2*cos(0.5)*a_j for j = 1..4. A
     torn read (half of one sample, half of the next) breaks that identity.
     On arm64 this is where a missing acquire/release would show up.
  3. Timestamps never go backwards, and they advance (the loop is live).
  4. Every pose this process reports as delivered is counted by the loop:
     the loop's own poses_received must equal the delivered count exactly.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import spatial_tools  # noqa: E402  (also puts cpp/python on sys.path)
from shared_ring_buffer import Pose6D  # noqa: E402

RECURRENCE = 2 * math.cos(0.5)
TOLERANCE = 1e-9  # doubles are copied bit-for-bit; this only absorbs the arithmetic


def wait_for_loop(timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        last = spatial_tools.read_joint_states.invoke({})
        if last.startswith("Current joint angles"):
            return
        spatial_tools._joint_bridge = None  # retry attaching if the segment wasn't there yet
        time.sleep(0.2)
    raise SystemExit(f"FAIL: motor_control_rt_loop never published telemetry: {last}")


def check_sample(angles) -> float:
    """Largest deviation from the sine recurrence across one sample."""
    return max(abs(angles[j - 1] + angles[j + 1] - RECURRENCE * angles[j]) for j in range(1, 5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path, help="the loop's --report path")
    parser.add_argument("--duration", type=float, default=15.0, help="seconds of reads and pushes")
    parser.add_argument("--loop-exit-timeout", type=float, default=60.0)
    args = parser.parse_args()

    # Negative control: the torn-read check must flag a sample stitched from
    # two periods, or a clean result would prove nothing.
    whole = [0.3 * math.sin(1.0 + 0.5 * j) for j in range(6)]
    later = [0.3 * math.sin(1.01 + 0.5 * j) for j in range(6)]  # one period on
    assert check_sample(whole) <= TOLERANCE, "check rejects a consistent sample"
    assert check_sample(whole[:3] + later[3:]) > TOLERANCE, "check misses a torn sample"

    wait_for_loop(timeout_s=20)

    # 1. The agent's tools, end to end.
    delivered = 0
    reply = spatial_tools.submit_spatial_intent.invoke({"tvec": (0.1, 0.2, 0.3), "rvec": (0.0, 0.0, 0.5)})
    if not reply.startswith("Spatial intent delivered"):
        raise SystemExit(f"FAIL: agent tool didn't deliver a pose: {reply}")
    delivered += 1
    print(f"agent tools OK: {spatial_tools.read_joint_states.invoke({})[:60]}...")

    # 2-4. Sustained reads and pushes through the tools' own bridge handles:
    # each ring is single-producer/single-consumer, so this process must stay
    # one reader of the joint ring and one writer of the pose ring.
    joints = spatial_tools._get_joint_bridge()
    poses = spatial_tools._get_pose_bridge()
    samples = torn = backwards = full = 0
    worst = 0.0
    first_ts = last_ts = None
    i = 0
    end = time.monotonic() + args.duration
    while time.monotonic() < end:
        state = joints.read_latest()
        if state is not None:
            samples += 1
            deviation = check_sample(state.joint_angles_rad)
            worst = max(worst, deviation)
            if deviation > TOLERANCE:
                torn += 1
            if last_ts is not None and state.timestamp_ns < last_ts:
                backwards += 1
            first_ts = state.timestamp_ns if first_ts is None else first_ts
            last_ts = state.timestamp_ns
        i += 1
        if i % 4 == 0:  # ~a few hundred poses per second, well under the loop's 1 kHz drain
            pose = Pose6D(tvec=(i * 1e-3, -i * 1e-3, 0.5), rvec=(0.0, 0.1, 0.2), timestamp_ns=time.monotonic_ns())
            if poses.push(pose):
                delivered += 1
            else:
                full += 1
        time.sleep(0.0005)

    advanced_ms = (last_ts - first_ts) / 1e6 if samples > 1 else 0.0
    print(f"joint samples read: {samples}, torn: {torn}, worst deviation: {worst:.3e}, "
          f"timestamps backwards: {backwards}, advanced: {advanced_ms:.0f} ms")
    print(f"poses delivered: {delivered}, rejected (ring full): {full}")

    # The loop writes its report when it exits; wait for it.
    deadline = time.monotonic() + args.loop_exit_timeout
    while not args.report.exists() and time.monotonic() < deadline:
        time.sleep(0.5)
    if not args.report.exists():
        raise SystemExit(f"FAIL: no report at {args.report}; did the loop exit?")
    time.sleep(0.5)  # let the writer finish
    report = json.loads(args.report.read_text())
    received = report["poses_received"]
    print(f"loop counted poses_received: {received}")

    failures = []
    if samples < 1000:
        failures.append(f"only {samples} joint samples read")
    if torn:
        failures.append(f"{torn} torn joint samples (worst deviation {worst:.3e})")
    if backwards:
        failures.append(f"{backwards} timestamps went backwards")
    if advanced_ms < args.duration * 1000 * 0.5:
        failures.append(f"timestamps advanced only {advanced_ms:.0f} ms over {args.duration:.0f} s")
    if received != delivered:
        failures.append(f"loop received {received} poses but {delivered} were delivered")
    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
