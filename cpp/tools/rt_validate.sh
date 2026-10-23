#!/usr/bin/env bash
# One-command real-time validation for the Agrios control loop.
#
# Runs cyclictest -- the standard PREEMPT_RT latency benchmark -- and the
# Agrios 1 kHz control loop on the same core, at the same SCHED_FIFO priority
# and period, under the same background stress load, and writes a
# side-by-side report (summary.md) plus the raw JSON from every run.
#
# Fairness, because a comparison is only worth something if it's even:
#   - the load runs for a warm-up period before anything is measured, so
#     neither tool measures the stressors' start-up transient;
#   - each tool's time is split into two runs, interleaved cyclictest, Agrios,
#     Agrios, cyclictest, so neither tool always gets the calmer slot;
#   - both run with every non-RT thread kept off the measured core
#     (cyclictest via --mainaffinity, the Agrios loop by design);
#   - compare_latency.py computes every statistic for both tools the same
#     way, from their merged 1 us histograms.
# The same script runs in CI (shared cloud VMs, stock kernel) and on real
# hardware with a PREEMPT_RT kernel; the report says which it was.
#
# Usage (from anywhere inside the repo):
#   cpp/tools/rt_validate.sh [--duration SECONDS] [--warmup SECONDS] [--label NAME]
#                            [--core N] [--priority P] [--out DIR] [--no-install]
#
#   --duration  measured seconds per tool (default 300 = 300,000 periods
#               each), split into two interleaved runs
#   --warmup    seconds of load before measuring starts (default 30)
#   --label     name for this machine in the report (default: this-machine).
#               Use something generic -- it ends up in a file you may publish.
#   --core      CPU core to pin both runs to (default: the last core)
#   --priority  SCHED_FIFO priority for both runs (default 90 -- see
#               cpp/README.md on why not 99)
#   --out       output directory (default: results/rt/<label>-<timestamp>)
#   --no-install  don't apt-get missing tools; fail instead
#
# Needs sudo: SCHED_FIFO, mlockall and /dev/cpu_dma_latency require root.
set -euo pipefail

DURATION=300
WARMUP=30
LABEL="this-machine"
CORE=""
PRIORITY=90
OUT=""
INSTALL=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration) DURATION="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --core) CORE="$2"; shift 2 ;;
    --priority) PRIORITY="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --no-install) INSTALL=0; shift ;;
    -h|--help) sed -n '2,39p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# ---- Tools -----------------------------------------------------------------
missing=()
for tool in cmake g++ cyclictest stress-ng python3; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  if [[ $INSTALL -eq 1 ]] && command -v apt-get >/dev/null 2>&1; then
    echo ">> installing: cmake g++ rt-tests stress-ng python3"
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake g++ rt-tests stress-ng python3 >/dev/null
  else
    echo "missing tools: ${missing[*]} (on Debian/Ubuntu: sudo apt-get install cmake g++ rt-tests stress-ng python3)" >&2
    exit 1
  fi
fi

# ---- Build -----------------------------------------------------------------
echo ">> building the control loop (Release)"
cmake -S cpp -B cpp/build-rt -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build cpp/build-rt -j --target motor_control_rt_loop >/dev/null

# ---- Machine facts ----------------------------------------------------------
NCPU="$(nproc)"
[[ -z "$CORE" ]] && CORE=$((NCPU - 1))
HALF_LOOPS=$((DURATION * 1000 / 2))
KREL="$(uname -r)"
PREEMPT_RT="no"
if [[ "$(cat /sys/kernel/realtime 2>/dev/null || echo 0)" == "1" ]] || uname -v | grep -q PREEMPT_RT; then
  PREEMPT_RT="yes"
fi
CPU_MODEL="$(grep -m1 -E 'model name|Model' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//' || true)"
[[ -z "$CPU_MODEL" ]] && CPU_MODEL="$(uname -m)"
GOVERNOR="$(cat /sys/devices/system/cpu/cpu"$CORE"/cpufreq/scaling_governor 2>/dev/null || echo "n/a")"
if grep -qw isolcpus /proc/cmdline; then
  ISOLATION="isolcpus set on the kernel command line"
else
  ISOLATION="none (no isolcpus/nohz_full -- other work can still run on the pinned core)"
fi

# Every CPU except the measured one, for cyclictest's main thread -- the same
# placement the Agrios loop gives its own non-RT threads.
OTHER_CPUS="$(seq 0 $((NCPU - 1)) | grep -vx "$CORE" | paste -sd, -)"
[[ -z "$OTHER_CPUS" ]] && OTHER_CPUS="$CORE"  # single-CPU machine: nowhere else to go

[[ -z "$OUT" ]] && OUT="results/rt/${LABEL}-$(date +%Y%m%d-%H%M)"
mkdir -p "$OUT"

echo ">> kernel $KREL, PREEMPT_RT: $PREEMPT_RT, $NCPU CPUs ($CPU_MODEL), core $CORE, governor $GOVERNOR"
echo ">> ${WARMUP}s load warm-up, then ${DURATION}s per tool in interleaved runs -- about $(((2 * DURATION + WARMUP) / 60 + 1)) minutes"

# ---- Load --------------------------------------------------------------------
# CPU hogs on every core (including the pinned one -- the point is that a
# SCHED_FIFO thread preempts them), memory churn, and filesystem syncs.
LOAD_DESC="stress-ng --cpu ${NCPU} --vm 2 --vm-bytes 20% --io 2 (all cores, including the pinned one)"
stress-ng --cpu "$NCPU" --vm 2 --vm-bytes 20% --io 2 --timeout "$((2 * DURATION + WARMUP + 120))s" --quiet &
STRESS_PID=$!
cleanup() { kill "$STRESS_PID" 2>/dev/null || true; wait "$STRESS_PID" 2>/dev/null || true; }
trap cleanup EXIT
echo ">> load warm-up (${WARMUP}s)"
sleep "$WARMUP"

run_cyclictest() {  # $1 = run number
  echo ">> cyclictest, run $1 (platform baseline)"
  sudo cyclictest --mlockall --priority="$PRIORITY" --interval=1000 --affinity="$CORE" \
    --mainaffinity="$OTHER_CPUS" --loops="$HALF_LOOPS" --quiet --histogram=10000 \
    --histfile="$OUT/cyclictest-$1-histogram.txt" --json="$OUT/cyclictest-$1.json" >/dev/null
}
run_agrios() {  # $1 = run number
  echo ">> Agrios control loop, run $1"
  sudo ./cpp/build-rt/motor_control_rt_loop --priority "$PRIORITY" --core "$CORE" \
    --iterations "$HALF_LOOPS" --quiet --report "$OUT/agrios-$1.json" | tail -n 2
}

# Interleaved: cyclictest, Agrios, Agrios, cyclictest.
run_cyclictest 1
run_agrios 1
run_agrios 2
run_cyclictest 2

cleanup
trap - EXIT
sudo chown -R "$(id -u):$(id -g)" "$OUT"

CONTEXT="Machine: ${CPU_MODEL}, ${NCPU} CPUs; CPU frequency governor: ${GOVERNOR}; core isolation: ${ISOLATION}. ${DURATION} s measured per tool, after a ${WARMUP} s load warm-up."
python3 cpp/tools/compare_latency.py \
  --cyclictest "$OUT/cyclictest-1.json" "$OUT/cyclictest-2.json" \
  --agrios "$OUT/agrios-1.json" "$OUT/agrios-2.json" \
  --label "$LABEL" --load "$LOAD_DESC" --context "$CONTEXT" --require-zero-faults | tee "$OUT/summary.md"

echo
echo ">> results saved in $OUT/ (summary.md plus the raw JSON from every run)"
