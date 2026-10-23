#!/usr/bin/env python3
"""Compare the Agrios 1 kHz control loop's wake-up latency with cyclictest's.

cyclictest (from rt-tests) is the standard latency benchmark for PREEMPT_RT
systems: a bare thread that sleeps to an absolute deadline, wakes, and records
how late it woke. The Agrios loop measures exactly the same quantity for its
own wake-ups (motor_control_rt_loop --report), while also doing its real
per-period work over the shared-memory bridge. Measured under the same
conditions, the gap between the two columns is what the Agrios loop adds on
top of what the platform itself delivers.

Fairness rules this script relies on (tools/rt_validate.sh sets them up):
  - several runs per tool, interleaved (cyclictest, Agrios, Agrios,
    cyclictest), so neither tool always gets the quieter time slot;
  - every statistic for both tools is computed here, the same way, from the
    merged 1 us histograms -- not taken from each tool's own summary.

Usage:
    compare_latency.py --cyclictest a.json [b.json ...] --agrios c.json [d.json ...] \
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
from collections import Counter


def stats_from_histogram(hist: Counter, overflow: int, exact_min: float, exact_max: float) -> dict:
    """Latency statistics from a {whole_us: count} histogram plus samples beyond its range.
    Percentiles: the smallest whole-microsecond value with at least q of the samples at or
    below it; the exact maximum when that sample lies beyond the histogram."""
    total = sum(hist.values()) + overflow

    def pct(q: float) -> float:
        target = max(1, math.ceil(q * total))
        cumulative = 0
        for us in sorted(hist):
            cumulative += hist[us]
            if cumulative >= target:
                return float(us)
        return float(exact_max)

    return {
        "samples": total,
        "min": float(exact_min),
        "p50": pct(0.50),
        "p99": pct(0.99),
        "p99_9": pct(0.999),
        "p99_99": pct(0.9999),
        "max": float(exact_max),
        "over_200us": sum(v for k, v in hist.items() if k > 200) + overflow,
        "missed": sum(v for k, v in hist.items() if k >= 1000) + overflow,
    }


def load_cyclictest(paths: list[str]) -> dict:
    hist, overflow, mins, maxes, weighted_mean, total = Counter(), 0, [], [], 0.0, 0
    preempt_rt, kernel = False, "?"
    per_run = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        threads = data["thread"]
        if len(threads) != 1:
            raise SystemExit(f"{path}: expected one measurement thread, found {len(threads)}")
        t = next(iter(threads.values()))
        h = Counter({int(k): int(v) for k, v in t.get("histogram", {}).items()})
        cycles = int(t["cycles"])
        hist += h
        run_overflow = max(0, cycles - sum(h.values()))  # samples beyond the histogram range
        overflow += run_overflow
        per_run.append(stats_from_histogram(h, run_overflow, float(t["min"]), float(t["max"])))
        mins.append(float(t["min"]))
        maxes.append(float(t["max"]))
        weighted_mean += float(t["avg"]) * cycles
        total += cycles
        preempt_rt |= bool(data.get("sysinfo", {}).get("realtime", 0))
        kernel = data.get("sysinfo", {}).get("release", kernel)
    out = stats_from_histogram(hist, overflow, min(mins), max(maxes))
    out.update(mean=weighted_mean / total if total else float("nan"), runs=len(paths),
               per_run=per_run, preempt_rt=preempt_rt, kernel=kernel)
    return out


def load_agrios(paths: list[str]) -> dict:
    hist, overflow, mins, maxes, weighted_mean, total = Counter(), 0, [], [], 0.0, 0
    iterations = dropped = minor = major = 0
    faults_measured, dma_held = True, True
    first = None
    per_run = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        first = first or data
        if "histogram_us" not in data:
            raise SystemExit(f"{path}: no histogram_us -- rebuild motor_control_rt_loop")
        h = Counter({int(k): int(v) for k, v in data["histogram_us"].items()})
        run_overflow = int(data.get("histogram_overflow", 0))
        hist += h
        overflow += run_overflow
        lat = data["latency_us"]
        per_run.append(stats_from_histogram(h, run_overflow, float(lat["min"]), float(lat["max"])))
        mins.append(float(lat["min"]))
        maxes.append(float(lat["max"]))
        weighted_mean += float(lat["mean"]) * int(data["samples"])
        total += int(data["samples"])
        iterations += int(data["iterations"])
        dropped += int(data["telemetry_dropped"])
        faults = data.get("page_faults_in_loop")
        if faults is None:
            faults_measured = False
        else:
            minor += int(faults["minor"])
            major += int(faults["major"])
        dma_held &= bool(data.get("cpu_dma_latency_held", False))
    out = stats_from_histogram(hist, overflow, min(mins), max(maxes))
    out.update(
        mean=weighted_mean / total if total else float("nan"), runs=len(paths), per_run=per_run,
        iterations=iterations, dropped=dropped,
        faults={"minor": minor, "major": major} if faults_measured else None,
        dma_latency_held=dma_held, preempt_rt=bool(first["kernel"]["preempt_rt"]),
        kernel=first["kernel"]["release"], priority=first.get("priority"), core=first.get("core"),
    )
    return out


def run_order(ct_runs: list, ag_runs: list) -> list:
    """(tool, stats) pairs in the order rt_validate.sh runs them: with two runs each that's
    cyclictest, Agrios, Agrios, cyclictest; otherwise all cyclictest runs, then all Agrios."""
    if len(ct_runs) == 2 and len(ag_runs) == 2:
        return [("cyclictest", ct_runs[0]), ("Agrios", ag_runs[0]),
                ("Agrios", ag_runs[1]), ("cyclictest", ct_runs[1])]
    return [("cyclictest", r) for r in ct_runs] + [("Agrios", r) for r in ag_runs]


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
        f"{ag['priority']}, pinned to core {ag['core']}, memory locked, every other thread kept off "
        f"that core. Load during all runs: {load}.",
    ]
    if context:
        lines += ["", context]
    lines += [
        "",
        f"Wake-up latency in microseconds (how late each period's wake-up was vs. its deadline), "
        f"{ct['runs']} cyclictest runs and {ag['runs']} Agrios runs, interleaved, merged; every "
        f"figure below is computed the same way for both, from the merged histograms:",
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
        "Each run on its own, in the order they ran (how much each tool varies from run to run "
        "is the yardstick for any gap between them):",
        "",
        "| run | tool | p50 | p99 | p99.9 | max | > 200 µs late |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, (tool, r) in enumerate(run_order(ct["per_run"], ag["per_run"]), start=1):
        lines.append(f"| {i} | {tool} | {fmt(r['p50'])} | {fmt(r['p99'])} | {fmt(r['p99_9'])} | "
                     f"{fmt(r['max'])} | {r['over_200us']:,} |")
    lines += [
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
    lines.append("- CPU kept out of deep idle states (/dev/cpu_dma_latency = 0, as cyclictest does): "
                 + ("yes" if ag["dma_latency_held"] else "no -- unavailable, so C-state exits may add latency"))
    lines += [
        "",
        f"Gap (Agrios − cyclictest): p99 {ag['p99'] - ct['p99']:+.0f} µs, "
        f"p99.9 {ag['p99_9'] - ct['p99_9']:+.0f} µs, max {ag['max'] - ct['max']:+.0f} µs. "
        "A gap of either sign at the tail is mostly the platform varying between runs: the loop "
        "can't wake faster than the kernel wakes it, so what to look for is whether its "
        "distribution tracks the baseline's.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cyclictest", required=True, nargs="+")
    ap.add_argument("--agrios", required=True, nargs="+")
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
