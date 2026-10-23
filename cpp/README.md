# Agrios bridge — lock-free SPSC ring buffer over POSIX shared memory

A single-producer/single-consumer lock-free ring buffer, built to carry 6-DoF
pose data between a Python perception process and a C++ motor-control
process across a POSIX shared-memory segment (`shm_open`/`mmap`) — no kernel
round-trip, no serialization, no lock, on the hot path between "perception
sees something" and "motor control reacts to it."

This exists to close a specific gap in the rest of Agrios and its sibling
projects: none of them touch physical hardware or a real-time control loop.
This is the first piece that's meant to eventually run on real robotics-class
hardware (a Raspberry Pi or Jetson), not just a laptop.

**Requires Linux or macOS.** POSIX shared memory doesn't exist on native
Windows. Build under WSL, a Linux container, or a real Linux/macOS host —
`cmake -S cpp -B cpp/build` will refuse to configure anywhere else, on
purpose, rather than silently producing a binary that can't work.

## Layout

```
include/agrios/
  spsc_ring_buffer.hpp   the lock-free algorithm itself (template, header-only)
  pose6d.hpp             pose payload: translation + Rodrigues rotation vector + timestamp
  joint_state.hpp        joint-telemetry payload: 6 joint angles + timestamp
  shared_ring_buffer.hpp POSIX shm_open/mmap wrapper: places a ring buffer in shared memory
  bridge_config.hpp      the one place channel names/capacities are defined, shared by every caller
  bridge_c_api.h         plain-C ABI over both channels -- see below
  rt_thread.hpp          PREEMPT_RT thread setup: mlockall, SCHED_FIFO, CPU affinity,
                         idle-state (cpu_dma_latency) guard -- see below
  control_tick.hpp       the per-period control step, [[clang::nonblocking]] -- shared by the
                         RT binary and the tests, so what's verified is what runs
  nonblocking.h          AGRIOS_NONBLOCKING: [[clang::nonblocking]] on Clang 20+, empty elsewhere
src/
  bridge_c_api.cpp       implementation of the C API, built as libagrios_bridge.so
  motor_control_consumer.cpp   demo: simple, non-real-time -- verifies the pose bridge itself
  motor_control_rt_loop.cpp    the real-time-compliant 1 kHz control loop -- see below
python/
  shared_ring_buffer.py  ctypes wrapper calling into libagrios_bridge.so (both channels)
  perception_producer_demo.py    demo: the Python "perception" side of the pose channel
  joint_state_reader_demo.py     demo: reads joint telemetry the RT loop publishes
tests/
  spsc_ring_buffer_test.cpp    concurrency correctness test (see below)
  rt_safety_test.cpp           drives the real control step over real shared memory with a live
                               producer; run under RealtimeSanitizer and ThreadSanitizer in CI
  nonblocking_negative_control.cpp   must FAIL to compile -- proves the static check is running
tools/
  rt_validate.sh               one command: cyclictest vs. the control loop, same core/priority/load
  compare_latency.py           the side-by-side report it writes (stdlib only)

../spatial_tools.py       LangGraph tools (submit_spatial_intent, read_joint_states)
                           wiring the Agrios agent itself to both channels -- see below
```

## Why a compiled C API instead of letting Python poke the shared bytes directly

`push()`/`pop()`'s correctness depends on real `std::memory_order_acquire`/
`release` fences, not just on reading and writing the right bytes at the
right offsets. Python's `struct`/`ctypes` modules have no memory-order-aware
atomics. A hand-rolled Python-side load/store would *look* correct and
*happen to work* on x86-64, because x86's strong memory model papers over
the missing fences — and would be a real, silent correctness bug on a
weakly-ordered target like the ARM64 Jetson/Raspberry Pi boards this is
ultimately meant to run on. `bridge_c_api.cpp` compiles to a small shared
library that both the C++ demo and `shared_ring_buffer.py` (via `ctypes`)
call into, so there's exactly one implementation of the protocol — the real
one — instead of a second, Python-side reimplementation that has to be kept
in sync with the first by hand and would only be caught by testing on the
right CPU architecture.

## The memory-ordering contract

- Producer: read `tail_` (own cursor) relaxed → check `head_` **acquire**
  (must see everything the consumer did before freeing this slot) → write
  the payload → publish by storing `tail_` **release** (so the payload write
  happens-before the consumer's acquire-load of `tail_` observes it).
- Consumer: mirror image — read `head_` relaxed → check `tail_` **acquire**
  (must see the payload the producer just wrote) → read the payload →
  publish the freed slot by storing `head_` **release**.

`head_`, `tail_`, and the payload buffer each start on their own 64-byte
(`alignas(kCacheLineSize)`) boundary, so a store to one by its owning thread
never invalidates the other thread's cached copy of the other index — the
textbook false-sharing failure mode for SPSC queues, avoided rather than
discovered later in a profiler. `kCacheLineSize` is a fixed constant, not
`std::hardware_destructive_interference_size`: GCC explicitly warns that
value "can vary between compiler versions or with different -mtune/-mcpu
flags" and recommends a fixed constant for anything ABI-facing — which this
is, since `bridge_c_api.cpp` and `shared_ring_buffer.py` both depend on the
resulting byte offsets staying stable.

The exact byte offsets this produces (`head_` at 0, `tail_` at 64, the
buffer at 128) are asserted at compile time in `spsc_ring_buffer.hpp` via
`static_assert(offsetof(...))` — not because the C++ side needs them, but
because `bridge_c_api.cpp`'s ABI needs to stay stable, and any future
reordering of the members would otherwise fail silently rather than at
build time.

## Verifying correctness

`tests/spsc_ring_buffer_test.cpp` runs a real producer thread against a real
consumer thread for 5,000,000 items through a deliberately small (256-slot)
buffer, forcing constant index wraparound, and checks for lost, duplicated,
or reordered items. A memory-ordering bug here would very plausibly pass a
single-threaded smoke test and only surface under real contention — so the
test is written to make that contention actually happen, and is meant to
also be run under ThreadSanitizer (see the comment at the top of the file)
for real happens-before-graph verification, not just a lucky pass.

`tests/rt_safety_test.cpp` is about the control loop, not the algorithm.
It drives the real `control_tick()` (the same function the RT binary runs)
over real `shm_open`/`mmap` bridges, with a producer thread pushing poses and
a telemetry drainer that stops halfway, so the no-pose, pose-received, and
telemetry-ring-full paths all execute. CI runs it three ways:

- **Under RealtimeSanitizer** (`-fsanitize=realtime`, Clang 20+). Any
  allocation, free, lock, or blocking call reached from `control_tick` aborts
  the run with a stack trace. `--negative-control` allocates inside a
  nonblocking function on purpose, and CI requires RealtimeSanitizer to kill it.
- **Under ThreadSanitizer**, covering the shared-memory bridge from several
  threads.
- **As a plain functional test** in every ordinary build (`ctest`).

The static half is a whole-project build with `clang++-20 -Werror
-Wfunction-effects`: anything `control_tick` calls that isn't provably
nonblocking fails the build, including the bridge's C API (annotated and
checked the same way). `nonblocking_negative_control.cpp` must fail that
check. If `AGRIOS_NONBLOCKING` ever silently expanded to nothing,
`control_tick` would keep compiling cleanly while checking nothing; this file
turns CI red instead.

## The real-time control loop (`motor_control_rt_loop.cpp`)

`motor_control_consumer.cpp` (above) is deliberately simple — a poll loop
with a sleep and a `std::cout` per pose, built to verify the bridge itself,
not to demonstrate real-time discipline. `motor_control_rt_loop.cpp` is the
actual PREEMPT_RT / ROS 2 real-time-guideline-compliant 1 kHz loop, built on
top of the same bridge:

- **`mlockall(MCL_CURRENT | MCL_FUTURE)` + stack pre-fault**, first thing in
  the RT thread's body (`agrios::rt::configure_current_thread_realtime` in
  `rt_thread.hpp`). `mlockall` stops any page of this process from being
  swapped out; it does *not* by itself guarantee a page is populated before
  its first touch, so the thread's stack is also explicitly pre-faulted
  (touched once, before the hot loop, using a stack sized for exactly this
  via `make_realtime_thread_attr`) rather than risking a first-touch minor
  fault from an unusually deep call inside the loop.
- **`SCHED_FIFO`, priority 99**, via `pthread_setschedparam`. Worth knowing
  before copying this onto real hardware: priority 99 is the same priority
  Linux gives several of its own kernel housekeeping threads (e.g. the
  per-CPU `migration/N` threads) — a runaway or misbehaving user thread at
  99 can starve kernel-internal work that the box needs to stay healthy.
  Real PREEMPT_RT deployments commonly reserve 99 and run application
  threads at something like 90-98 for exactly this reason. This code uses
  99 because that's what was specified; know what you're asking for if you
  reuse it.
- **`pthread_setaffinity_np()`**, pinning to one CPU core. Pinning a thread
  to a core is not the same thing as isolating that core: without also
  booting the kernel with `isolcpus=`, `nohz_full=`, and `rcu_nocbs=` for
  that core, the scheduler is still free to run other tasks there between
  this thread's slices, and periodic kernel work (timer ticks, RCU
  callbacks) can still land on it. That kernel-boot-parameter configuration
  is a host-level prerequisite this code cannot set up or verify from
  inside a single process — treat "pinned" and "isolated" as two different
  claims.
- **Zero dynamic allocation and no `std::cout` inside the hot loop.** Every
  loop-body value is a stack value or a pre-allocated buffer. Telemetry
  (per-iteration jitter, pose-received flag) is pushed into a second,
  in-process `SPSCRingBuffer` — the same primitive verified above, reused
  for a thread-to-thread pairing instead of the cross-process one — and
  drained by a separate, ordinary-priority monitor thread that does all the
  actual printing. This is the same shape as ROS 2 control's
  `RealtimeBuffer`/`RealtimePublisher`: get state out of the RT loop without
  letting a logger's I/O jitter leak into the control loop's own timing.
- **Absolute-deadline scheduling**, not a relative sleep. Each iteration
  advances a `CLOCK_MONOTONIC` deadline by exactly one period and sleeps to
  that deadline with `clock_nanosleep(..., TIMER_ABSTIME, ...)`. A relative
  `sleep_for(period)` re-measured from "now" each pass would accumulate that
  pass's own execution time as drift on every iteration; anchoring to a
  fixed, monotonically-advancing deadline doesn't.
- **`-Wl,-z,now`** on this executable specifically (see `CMakeLists.txt`):
  forces eager, load-time symbol resolution instead of lazy binding, so the
  *first* call to a dynamically-linked function from inside the hot loop
  doesn't pay for on-demand dynamic-linker resolution — another possible
  source of first-touch, hard-to-reproduce latency this loop exists to
  eliminate.
- **One clock read during setup.** `clock_gettime()` runs from the vDSO,
  which is set up lazily and isn't covered by `mlockall`. Before this warm-up
  existed, the RT thread's measured page-fault count was exactly 1 on every
  run, all from the loop's first clock read. It's now 0.
- **`CpuDmaLatencyGuard`**: `/dev/cpu_dma_latency` held at 0 for the whole
  run, so the CPU stays out of deep idle states whose exit latency would
  otherwise land on every wake-up. `cyclictest` does the same by default,
  which is what makes comparing against it fair. Needs root; if it's
  unavailable, the run continues and the report records it.
- **`keep_current_thread_off_core()`**: `main` moves itself off the RT core
  before creating any thread, so the monitor thread inherits that and never
  runs where the RT thread needs to wake. The RT thread pins itself onto the
  core during its own setup.
- **`--report results.json`**: wake-up latency percentiles (1 µs bins, the
  resolution `cyclictest` reports), overruns, missed periods, telemetry
  samples lost, page faults the RT thread took inside the loop, whether the
  idle-state guard was held, and the kernel (including whether it's
  PREEMPT_RT). The telemetry ring holds 4096 samples (about 4 s), so a
  monitor thread starved by heavy load doesn't lose any; any that are lost
  are counted, not hidden.

### What was actually verified, and what wasn't

Built and run in the same Linux container as everything else above, granted
`--cap-add=SYS_NICE --cap-add=IPC_LOCK` (`SCHED_FIFO` and `mlockall` both
need `CAP_SYS_NICE`/`CAP_IPC_LOCK` or root, which Docker does not grant by
default even to a root user inside the container):

- **Structural correctness: verified.** `mlockall`, `pthread_setschedparam(SCHED_FIFO, 99)`,
  and `pthread_setaffinity_np` all succeeded with the capabilities granted,
  and the loop ran its requested iteration count cleanly with zero crashes.
  Re-ran *without* those capabilities to confirm the failure path actually
  fires rather than silently degrading: it correctly reported
  `pthread_setschedparam(SCHED_FIFO, 99) failed: Operation not permitted`
  and exited nonzero without ever entering the loop, exactly as designed —
  a control loop that silently continues without the real-time guarantees
  it asked for and didn't get is worse than one that refuses to run.
- **Loop-timing logic: verified.** The absolute-deadline scheduling
  correctly produced the requested iteration count in the expected wall-clock
  time, with no drift accumulation, across a 2,000-iteration (~2 s) run.
- **Latency/jitter numbers (historical): measured, but not representative
  of anything beyond this dev machine.** Superseded by the side-by-side
  `cyclictest` measurement below, kept here as the record of what was
  measured at the time. That same run showed a 76 μs average and
  450 μs worst-case deadline miss, with 76/2000 iterations over a 200 μs
  overrun threshold. Those numbers are **not** a real-time performance claim
  — they were measured inside Docker Desktop on WSL2, i.e. a container on
  top of a *stock*, non-PREEMPT_RT-patched Linux kernel, itself inside a
  virtualized VM, with no core isolation configured anywhere in that stack.
  Every layer of that is exactly what PREEMPT_RT and `isolcpus` exist to
  eliminate. What's genuinely proven here is that the code's real-time
  *mechanisms* work correctly when granted the right privileges and that its
  *timing logic* doesn't drift — not that this loop meets any particular
  latency bound. Re-measuring on a real PREEMPT_RT-patched kernel with an
  actually-isolated core is a prerequisite for treating any latency number
  from this loop as meaningful.

## Measuring latency on real hardware

Real-time *safety* (no allocation, locks, blocking, or page faults inside
the loop) doesn't depend on the machine, and CI verifies it. An absolute
worst-case latency does: it belongs to a specific machine's kernel,
firmware, and core isolation. So it's measured per machine, against
`cyclictest` on that same machine, by one script:

```bash
cpp/tools/rt_validate.sh --duration 300 --label my-laptop
```

The script:
1. Installs `cmake g++ rt-tests stress-ng` if they're missing (Debian/Ubuntu).
2. Builds the loop.
3. Starts `stress-ng` on every core, the pinned one included, and lets it
   warm up (30 s by default) so neither tool measures its start-up spike.
4. Runs `cyclictest` and the Agrios loop at the same SCHED_FIFO priority (90
   by default), period (1 ms), and core. Each tool's time is split into two
   runs, interleaved: cyclictest, Agrios, Agrios, cyclictest. `cyclictest`
   gets `--mainaffinity` so its main thread stays off the measured core, as
   the Agrios loop's non-RT threads do.
5. Writes `results/rt/<label>-<time>/summary.md` with both columns, plus the
   raw JSON from every run. Every statistic for both tools is computed by
   `compare_latency.py` the same way, from the merged 1 µs histograms.

It reports whether the kernel was PREEMPT_RT, the CPU, the frequency
governor, and whether any core isolation was configured, so a number is
never separated from the conditions it was measured under. It exits nonzero
if the RT thread took a page fault inside its loop.

**Without installing anything:** most laptops and desktops can boot
LinuxCNC's live USB image. LinuxCNC is a machine-control project, so its
image ships a PREEMPT_RT kernel and runs entirely from the stick. Write the
image to a USB drive with balenaEtcher or Rufus, boot from it using the
firmware's one-time boot menu (F12 on most Dell machines), choose the live
session (not "Install"), connect to the network, then:

```bash
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/Zautzik/agrios && cd agrios
cpp/tools/rt_validate.sh --duration 300 --label <machine-name>
```

Copy `results/rt/` off the machine before rebooting, to a second USB drive
or by uploading it; a live session keeps nothing. Laptops are a demanding
real-time target (firmware SMIs, power management), so treat the result as a
measurement of that machine, not a ceiling for the code.

## Two channels, and the pose representation

There are two independent shared-memory channels, not one bidirectional
one, because they're two genuinely separate producer/consumer pairs and
`SPSCRingBuffer` is single-producer/single-consumer by design:

- `/agrios_pose_bridge` (`Pose6D`) — perception → motor control. Translation
  vector + Rodrigues rotation vector (`tvec`/`rvec`), not Euler angles: this
  is what `cv2.solvePnP`, ArUco pose estimation, and FoundationPose all
  return directly, so a perception pipeline can push its output here with
  zero conversion, and it's free of gimbal lock (there's no Euler axis order
  that avoids all degenerate orientations; a rotation vector has no such
  case). `Pose6D` originally stored Euler angles (`x, y, z, pitch, yaw,
  roll`) as a placeholder from before this had a real producer in mind —
  migrated to `tvec`/`rvec` once that placeholder needed to accept real
  perception-pipeline output, including rebuilding and re-verifying
  everything that already depended on the old layout (see NOTES.md).
- `/agrios_joint_telemetry` (`JointState6`) — motor control → monitoring/
  perception, the opposite direction. Six joint angles (radians) + a
  timestamp. There's no real hardware behind this yet, so
  `motor_control_rt_loop.cpp` publishes a clearly-synthetic waveform each
  iteration — what's real is the non-blocking `push()` pattern being
  exercised, not the numbers.

## LangGraph integration (`../spatial_tools.py`)

Two tools wire the Agrios agent itself to both channels, kept in their own
module rather than alongside the fixture/SQLAlchemy tools in `main.py`
because this module's dependency footprint (POSIX shared memory via
`ctypes`, a sibling C++ build) is fundamentally different from the rest of
the agent:

- **`submit_spatial_intent`** — a `SubmitSpatialIntent` Pydantic model
  (`tvec`, `rvec`, each a fixed 3-tuple of floats) as the tool's
  `args_schema`, pushing the validated pose onto the pose channel. Its
  return value distinguishes "delivered to the ring buffer" from "no
  motor-control process is attached to receive it" — `push()` succeeding
  only means there was room in the buffer, not that anything is reading it.
- **`read_joint_states`** — no arguments; reads the freshest joint-telemetry
  sample via `JointStateBridge.read_latest()`, which drains the ring buffer
  to the newest entry rather than returning whatever's oldest (right for
  "what's the current state," wrong for the pose channel where every pose
  matters) — still non-blocking, since each individual `pop()` inside that
  drain is itself wait-free.

Both fail gracefully rather than crashing the agent when the C++ side isn't
available: `libagrios_bridge.so` not being built, or no motor-control
process having created the shared-memory segment yet, both come back as a
plain string result the LLM can read, not an exception. This matters
concretely on this project's own Windows dev machine, where the bridge
can't exist at all (POSIX shared memory), and the two tools need to degrade
to "not available right now" instead of taking the whole agent down at
import time — verified directly: `uv run python -c "import main"` and
invoking both tools succeed on Windows with no C++ side present, returning
a clear unavailability message.

**The full loop was verified for real**, inside the Linux container, not
just each half in isolation: installed `pydantic`+`langchain-core`, started
a real `motor_control_rt_loop` process, then from a separate Python process
imported the actual `spatial_tools` module and called the actual
`@tool`-decorated functions (not a mock, not a unit test double) —
`submit_spatial_intent.invoke({"tvec": [0.15, -0.2, 0.4], "rvec": [0.0, 0.0,
0.785]})` reported delivery, and the RT loop's own telemetry log confirmed
it: `pose=no` on every line until the exact iteration the push landed, then
`pose=yes` from there on. `read_joint_states.invoke({})` then read back a
live sample from that same RT loop's synthetic telemetry. Every stage of
the requested architecture — Pydantic validation → LangGraph tool →
compiled C ABI → real-time thread, and back — was exercised end to end in
one run, not asserted from reading the code.

## Building and running the demos

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure   # runs spsc_ring_buffer_test

# Terminal 1 — C++ motor control (creates the shared-memory segment)
./cpp/build/motor_control_consumer

# Terminal 2 — Python perception
python cpp/python/perception_producer_demo.py --count 20
```

The real-time loop, in place of the simple consumer above (needs
`CAP_SYS_NICE`/`CAP_IPC_LOCK` or root — e.g. `docker run --cap-add=SYS_NICE
--cap-add=IPC_LOCK ...`):

```bash
./cpp/build/motor_control_rt_loop --core 0 --priority 99 --iterations 2000
```

Reading the RT loop's joint-telemetry channel from Python, in place of/
alongside the pose demo above:

```bash
python cpp/python/joint_state_reader_demo.py --count 10
```

Or exercise the actual LangGraph tools directly (needs `pydantic` and
`langchain-core`, and `libagrios_bridge.so` discoverable — see
`AGRIOS_BRIDGE_LIB` in `shared_ring_buffer.py` if it's not in one of the
default build-output locations):

```python
from spatial_tools import submit_spatial_intent, read_joint_states

submit_spatial_intent.invoke({"tvec": [0.15, -0.2, 0.4], "rvec": [0.0, 0.0, 0.785]})
read_joint_states.invoke({})
```
