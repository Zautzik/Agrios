#!/usr/bin/env bash
# One-command real-time validation for the Agrios control loop.
#
# Runs cyclictest -- the standard PREEMPT_RT latency benchmark -- and the
# Agrios 1 kHz control loop back to back, on the same core, at the same
# SCHED_FIFO priority and period, under the same background stress load, and
# writes a side-by-side report (summary.md) plus the raw JSON from both.
# The same script runs in CI (shared cloud VMs, stock kernel) and on real
# hardware with a PREEMPT_RT kernel; the report says which it was.
#
# Usage (from anywhere inside the repo):
#   cpp/tools/rt_validate.sh [--duration SECONDS] [--label NAME] [--core N]
#                            [--priority P] [--out DIR] [--no-install]
#
#   --duration  seconds per run (default 300 = 300,000 periods each)
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
LABEL="this-machine"
CORE=""
PRIORITY=90
OUT=""
INSTALL=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration) DURATION="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --core) CORE="$2"; shift 2 ;;
    --priority) PRIORITY="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --no-install) INSTALL=0; shift ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
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
LOOPS=$((DURATION * 1000))
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

[[ -z "$OUT" ]] && OUT="results/rt/${LABEL}-$(date +%Y%m%d-%H%M)"
mkdir -p "$OUT"

echo ">> kernel $KREL, PREEMPT_RT: $PREEMPT_RT, $NCPU CPUs ($CPU_MODEL), core $CORE, governor $GOVERNOR"
echo ">> two runs of ${DURATION}s each, under load -- about $((2 * DURATION / 60 + 1)) minutes"

# ---- Load --------------------------------------------------------------------
# CPU hogs on every core (including the pinned one -- the point is that a
# SCHED_FIFO thread preempts them), memory churn, and filesystem syncs.
LOAD_DESC="stress-ng --cpu ${NCPU} --vm 2 --vm-bytes 20% --io 2 (all cores, including the pinned one)"
stress-ng --cpu "$NCPU" --vm 2 --vm-bytes 20% --io 2 --timeout "$((2 * DURATION + 120))s" --quiet &
STRESS_PID=$!
cleanup() { kill "$STRESS_PID" 2>/dev/null || true; wait "$STRESS_PID" 2>/dev/null || true; }
trap cleanup EXIT
sleep 5

# ---- Baseline: cyclictest -------------------------------------------------------
echo ">> [1/2] cyclictest (platform baseline)"
sudo cyclictest --mlockall --priority="$PRIORITY" --interval=1000 --affinity="$CORE" \
  --loops="$LOOPS" --quiet --histogram=10000 --json="$OUT/cyclictest.json" > "$OUT/cyclictest-histogram.txt"

# ---- Agrios control loop ------------------------------------------------------
echo ">> [2/2] Agrios control loop"
sudo ./cpp/build-rt/motor_control_rt_loop --priority "$PRIORITY" --core "$CORE" \
  --iterations "$LOOPS" --quiet --report "$OUT/agrios.json" | tail -n 2

cleanup
trap - EXIT
sudo chown -R "$(id -u):$(id -g)" "$OUT"

CONTEXT="Machine: ${CPU_MODEL}, ${NCPU} CPUs; CPU frequency governor: ${GOVERNOR}; core isolation: ${ISOLATION}. ${DURATION} s per run."
python3 cpp/tools/compare_latency.py --cyclictest "$OUT/cyclictest.json" --agrios "$OUT/agrios.json" \
  --label "$LABEL" --load "$LOAD_DESC" --context "$CONTEXT" --require-zero-faults | tee "$OUT/summary.md"

echo
echo ">> results saved in $OUT/ (summary.md, agrios.json, cyclictest.json)"
