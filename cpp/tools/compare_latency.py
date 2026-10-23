#!/usr/bin/env python3
"""Compare the Agrios 1 kHz control loop's wake-up latency with cyclictest's.

cyclictest (from rt-tests) is the standard latency benchmark for PREEMPT_RT
systems: a bare thread that sleeps to an absolute deadline, wakes, and records
how late it woke. The Agrios loop measures exactly the same quantity for its
own wake-ups (motor_control_rt_loop --report), while also doing its real
per-period work over the shared-memory bridge. Run back to back on the same
core, priority, period and load, the gap between the two columns is what the
Agrios loop adds on top of what the platform itself delivers.

Usage:
    compare_latency.py --cyclictest cyclictest.json --agrios agrios.json \
        [--label NAME] [--load DESCRIPTION] [--context TEXT] [--require-zero-faults]

Prints a Markdown report. With --require-zero-faults, exits 1 if the Agrios
RT thread took any page fault inside its loop (mlockall + pre-faulting are
supposed to make that zero, so it's checked rather than assumed).
Standard library only.
"""
import argparse
import json
import math
import sys


def histogram_percentile(hist: dict[int, int], total: int, q: float, max_value: float) -> float:
    """Smallest whole-microsecond value with at least q of the samples at or below it.
    Falls back to the exact maximum when the sample lands beyond the histogram."""
    if total == 0:
        return float("nan")
    target = max(1, math.ceil(q * total))
    cumulative = 0
    for us in sorted(hist):
        cumulative += hist[us]
        if cumulative >= target:
            return float(us)
    return float(max_value)


def load_cyclictest(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    threads = data["thread"]
    if len(threads) != 1:
        raise SystemExit(f"{path}: expected one measurement thread, found {len(threads)}")
    t = next(iter(threads.values()))
    hist = {int(k): int(v) for k, v in t.get("histogram", {}).items()}
    total = int(t["cycles"])
    in_hist = sum(hist.values())
    beyond = max(0, total - in_hist)  # samples past the histogram range
    return {
        "samples": total,
        "min": float(t["min"]),
        "mean": float(t["avg"]),
        "p50": histogram_percentile(hist, total, 0.50, t["max"]),
        "p99": histogram_percentile(hist, total, 0.99, t["max"]),
        "p99_9": histogram_percentile(hist, total, 0.999, t["max"]),
        "p99_99": histogram_percentile(hist, total, 0.9999, t["max"]),
        "max": float(t["max"]),
        "over_200us": sum(v for k, v in hist.items() if k > 200) + beyond,
        "missed": sum(v for k, v in hist.items() if k >= 1000) + beyond,
        "preempt_rt": bool(data.get("sysinfo", {}).get("realtime", 0)),
        "kernel": data.get("sysinfo", {}).get("release", "?"),
    }


def load_agrios(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    lat = data["latency_us"]
    over_key = next(k for k in data if k.startswith("overruns_over_"))
    return {
        "samples": int(data["samples"]),
        "min": float(lat["min"]),
        "mean": float(lat["mean"]),
        "p50": float(lat["p50"]),
        "p99": float(lat["p99"]),
        "p99_9": float(lat["p99_9"]),
        "p99_99": float(lat["p99_99"]),
        "max": float(lat["max"]),
        "over_200us": int(data[over_key]),
        "missed": int(data["missed_periods"]),
        "iterations": int(data["iterations"]),
        "dropped": int(data["telemetry_dropped"]),
        "faults": data.get("page_faults_in_loop"),
        "preempt_rt": bool(data["kernel"]["preempt_rt"]),
        "kernel": data["kernel"]["release"],
        "priority": data.get("priority"),
        "core": data.get("core"),
        "dma_latency_held": data.get("cpu_dma_latency_held"),
    }


def fmt(value: float, decimals: int = 0) -> str:
    if value != value:  # NaN
        return "n/a"
    return f"{value:,.{decimals}f}"


def render(ct: dict, ag: dict, label: str, load: str, context: str) -> str:
    rt = "yes" if (ct["preempt_rt"] or ag["preempt_rt"]) else "no (stock kernel)"
    lines = [
        f"## Real-time latency: {label}",
        "",
        f"Kernel `{ag['kernel']}`, PREEMPT_RT: **{rt}**. Period 1 ms (1 kHz), SCHED_FIFO priority "
        f"{ag['priority']}, pinned to core {ag['core']}, memory locked. Load during both runs: {load}.",
    ]
    if context:
        lines += ["", context]
    lines += [
        "",
        "Wake-up latency, microseconds (how late each period's wake-up was vs. its deadline):",
        "",
        "| | cyclictest (platform baseline) | Agrios control loop |",
        "|---|---:|---:|",
        f"| samples | {ct['samples']:,} | {ag['samples']:,} |",
        f"| min | {fmt(ct['min'])} | {fmt(ag['min'], 1)} |",
        f"| mean | {fmt(ct['mean'], 1)} | {fmt(ag['mean'], 1)} |",
        f"| p50 | {fmt(ct['p50'])} | {fmt(ag['p50'])} |",
        f"| p99 | {fmt(ct['p99'])} | {fmt(ag['p99'])} |",
        f"| p99.9 | {fmt(ct['p99_9'])} | {fmt(ag['p99_9'])} |",
        f"| p99.99 | {fmt(ct['p99_99'])} | {fmt(ag['p99_99'])} |",
        f"| **max** | **{fmt(ct['max'])}** | **{fmt(ag['max'], 1)}** |",
        f"| wake-ups > 200 µs late | {ct['over_200us']:,} | {ag['over_200us']:,} |",
        f"| wake-ups ≥ 1 period late | {ct['missed']:,} | {ag['missed']:,} |",
        "",
        "Agrios-only checks:",
        "",
    ]
    faults = ag["faults"]
    if faults is None:
        lines.append("- Page faults in the RT loop: not measurable on this platform")
    else:
        lines.append(f"- Page faults taken by the RT thread inside the loop: **{faults['minor']} minor, "
                     f"{faults['major']} major**")
    lines.append(f"- Telemetry samples lost (monitor thread fell behind): {ag['dropped']:,} of "
                 f"{ag['iterations']:,}")
    held = ag["dma_latency_held"]
    if held is not None:
        lines.append("- CPU kept out of deep idle states (/dev/cpu_dma_latency = 0, as cyclictest does): "
                     + ("yes" if held else "no -- unavailable, so C-state exits may add latency"))
    lines += [
        "",
        f"Gap (Agrios − cyclictest): p99 {ag['p99'] - ct['p99']:+.0f} µs, "
        f"p99.9 {ag['p99_9'] - ct['p99_9']:+.0f} µs, max {ag['max'] - ct['max']:+.0f} µs. "
        "The two runs are sequential, not simultaneous, so some of any gap is run-to-run "
        "variation of the platform itself.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cyclictest", required=True)
    ap.add_argument("--agrios", required=True)
    ap.add_argument("--label", default="this machine")
    ap.add_argument("--load", default="none")
    ap.add_argument("--context", default="")
    ap.add_argument("--require-zero-faults", action="store_true")
    args = ap.parse_args()

    ct = load_cyclictest(args.cyclictest)
    ag = load_agrios(args.agrios)
    sys.stdout.write(render(ct, ag, args.label, args.load, args.context))

    if args.require_zero_faults:
        faults = ag["faults"]
        if faults is not None and (faults["minor"] or faults["major"]):
            print(f"\nFAIL: the RT thread took page faults inside its loop: {faults}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
