# Agrios

[![CI](https://github.com/Zautzik/Agrios/actions/workflows/ci.yml/badge.svg)](https://github.com/Zautzik/Agrios/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

**A lock-free, zero-copy POSIX shared-memory bridge between a Python LangGraph agent and a 1 kHz C++ real-time control loop, with its real-time claims checked rather than asserted.** The per-period control step is verified never to allocate, lock, or block: at compile time by Clang's function-effect analysis, and at run time by LLVM's RealtimeSanitizer. The bridge is race-free under ThreadSanitizer, the loop takes zero page faults, and its wake-up latency is measured against `cyclictest`, the standard real-time benchmark. All of it runs in CI on x86-64 and arm64, and every check has a negative control that must fail. Exact scope: [What's verified, and what isn't](#whats-verified-and-what-isnt).

## Architecture

```mermaid
flowchart LR
    subgraph PY["Python process — LangGraph agent (main.py)"]
        LLM["qwen2.5:7b (Ollama)<br/>ReAct tool-calling loop"]
        T1["submit_spatial_intent<br/>Pydantic-validated tvec/rvec"]
        T2["read_joint_states"]
        LLM -->|tool call| T1
        LLM -->|tool call| T2
    end

    subgraph SHM["POSIX shared memory — /dev/shm, zero-copy, lock-free"]
        POSE[("/agrios_pose_bridge<br/>SPSC ring buffer · Pose6D")]
        JOINT[("/agrios_joint_telemetry<br/>SPSC ring buffer · JointState6")]
    end

    subgraph CPP["C++ process — motor_control_rt_loop (no Python in this process)"]
        RT["RT thread<br/>SCHED_FIFO 99 · pinned core · mlockall<br/>1 kHz clock_nanosleep(TIMER_ABSTIME)<br/>zero allocation, zero std::cout"]
        TELE[("in-process SPSC ring buffer<br/>jitter / overrun samples")]
        MON["Monitor thread<br/>SCHED_OTHER — owns all I/O"]
        RT -->|non-blocking push| TELE -->|non-blocking pop| MON
    end

    T1 -->|"ctypes -> libagrios_bridge.so<br/>push(), release store"| POSE
    POSE -->|"non-blocking pop, every 1 ms tick<br/>acquire load"| RT
    RT -->|non-blocking push, synthetic telemetry| JOINT
    JOINT -->|"ctypes -> libagrios_bridge.so<br/>read_latest(), drains to newest"| T2
```

The two sides are **fully asynchronous, coupled only through shared memory** — neither ever calls into or blocks on the other. The Python side runs at whatever cadence the LLM decides to call a tool (irregular, on the order of seconds); the C++ side runs at a fixed 1 kHz forever, regardless of whether anything is reading or writing on the other end. `push()`/`pop()` on both channels are `O(1)` and non-blocking by construction (see [The IPC layer](#the-ipc-layer)), so a slow or absent Python process can never stall the control loop, and a slow or absent control loop can never hang a tool call — a full ring buffer just means `push()` returns `false` and the caller decides what to do, not a block.

## The IPC layer

- **POSIX shared memory (`shm_open`/`mmap`)**, not a socket, pipe, or serialized message format — the payload is placed directly at a shared virtual address both processes map, so there is no copy, no serialization/deserialization step, and no kernel round-trip on the hot path.
- **A C++20 lock-free single-producer/single-consumer ring buffer** (template, header-only) sits on top of that raw memory, with explicit `std::memory_order_acquire`/`release` fences — not a mutex, and not the default sequentially-consistent ordering — pairing the producer's release-store of its write index with the consumer's acquire-load of that same index, so a payload write always happens-before the consumer observes it, with no lock in either direction.
- **The control loop never executes inside a Python interpreter at all.** `motor_control_rt_loop` is a separate OS process, built as native C++ — there's no GIL to hold, no reference counting, no interpreter-level nondeterminism anywhere in the timing-critical path, because CPython simply isn't present in that process. Python only ever touches the bridge through a compiled C ABI (`libagrios_bridge.so`) called via `ctypes`, so both languages execute the *exact same* atomic operations — one implementation of the protocol, not a hand-rolled Python-side reimplementation that would only happen to work on x86-64's strong memory model and silently break on a weakly-ordered target like an ARM64 Jetson or Raspberry Pi.
- **Two independent channels, not one bidirectional one** — `Pose6D` (perception → control: translation + Rodrigues rotation vector, the representation `cv2.solvePnP`/ArUco/FoundationPose emit natively) and `JointState6` (control → monitoring: six joint angles). `SPSCRingBuffer` is single-producer/single-consumer by design, so multiplexing two unrelated data flows onto one channel would break that on the first mixed read.
- **The producer side is model-agnostic.** `submit_spatial_intent` doesn't care whether the `tvec`/`rvec` it validates came from an LLM tool call, a classical ArUco marker detector, or a learned VLA policy — the bridge's contract is the Pydantic schema and the shared-memory layout, not the thing generating the numbers. Today that producer is a LangGraph agent reasoning in natural language (see below); swapping in a real perception pipeline changes nothing on the C++ side.

Full design writeup — the memory-ordering contract, the exact byte-offset layout, why a compiled C ABI instead of raw `ctypes` struct-poking — in [cpp/README.md](cpp/README.md).

## Real-time hardening

`motor_control_rt_loop.cpp`'s RT thread, first thing in its body, before touching anything else:

- **`mlockall(MCL_CURRENT | MCL_FUTURE)`**, plus an explicit stack pre-fault — `mlockall` stops pages from being swapped out, but doesn't by itself guarantee a page is *populated* before its first touch, so the thread's stack is also touched once, up front, using a stack explicitly sized for it.
- **`SCHED_FIFO`, priority 99** via `pthread_setschedparam`. Worth knowing rather than glossing over: 99 is the same priority Linux gives several of its own kernel housekeeping threads (e.g. the per-CPU `migration/N` threads) — a misbehaving thread at 99 can starve kernel-internal work the box needs to stay healthy. Real deployments commonly reserve 99 and run application threads at 90–98 for exactly this reason; this code uses 99 because that's the spec, not because it's the recommended default.
- **`pthread_setaffinity_np()`**, pinned to a chosen core. Pinning is not isolation: without also booting the kernel with `isolcpus=`, `nohz_full=`, and `rcu_nocbs=` for that core, the scheduler is still free to run other work there between this thread's slices. That kernel-boot-parameter configuration is a host-level prerequisite this code cannot set up or verify from inside a single process — "pinned" and "isolated" are two different, easily-conflated claims.
- **Zero dynamic allocation and no `std::cout` inside the hot loop.** Every loop-body value is a stack value or a pre-allocated buffer; telemetry exits through the same lock-free ring-buffer primitive, reused for an in-process producer/consumer pair, drained by a separate ordinary-priority thread that owns all the actual I/O — the same shape as ROS 2 control's `RealtimeBuffer`/`RealtimePublisher`.
- **Absolute-deadline scheduling** (`clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, ...)`), not a relative sleep — a relative `sleep_for(period)` re-measured from "now" each pass accumulates that pass's own execution time as drift on every iteration; a fixed, monotonically-advancing deadline doesn't.
- **`-Wl,-z,now`** on this executable specifically, forcing eager symbol binding — without it, the *first* call to a dynamically-linked function from inside the hot loop pays for on-demand dynamic-linker resolution, another source of first-touch, hard-to-reproduce latency.
- **A clock read during setup, before the loop.** The vDSO mapping `clock_gettime()` runs from is set up lazily and isn't covered by `mlockall`. Measured, not guessed: without this warm-up, every run took exactly one page fault inside the loop, on its first clock read.
- **The CPU held out of deep idle states** (`/dev/cpu_dma_latency` = 0 for the whole run). Waking from a deep C-state can add tens to hundreds of microseconds on its own, a hardware latency that `SCHED_FIFO` doesn't touch. `cyclictest` does the same by default, which is also what makes comparing the two fair.
- **Every other thread kept off the RT core.** `main` moves itself, and so everything it spawns, off the pinned core before creating any thread; only the RT thread runs there.
- **The control step is a single `[[clang::nonblocking]]` function** (`control_tick.hpp`), shared by the binary and the tests. That turns "no allocation, no locks, no I/O" from a comment into something the compiler and a sanitizer check (below).

## What's verified, and what isn't

| Property | How it's checked | Where |
|---|---|---|
| Control step never allocates, locks, or blocks | Clang 20 function-effect analysis on `[[clang::nonblocking]]`, whole project at `-Werror`, plus a violation that must fail to compile | CI, x86-64 + arm64 |
| …and doesn't at run time | LLVM RealtimeSanitizer: a test driving the real control step with a live producer, plus the real RT binary for 5,000 periods at `SCHED_FIFO`; an allocation must be caught | CI, x86-64 + arm64 |
| No data races | ThreadSanitizer: 5,000,000-op SPSC stress test, plus the shared-memory bridge driven from several threads | CI, x86-64 + arm64 |
| Zero page faults inside the RT loop | `getrusage(RUSAGE_THREAD)` before and after the loop; CI fails on anything but zero | CI, any Linux machine |
| Wake-up latency | Side by side with `cyclictest`: same core, priority, period, and `stress-ng` load; interleaved runs after a warm-up; both computed the same way | CI (stock-kernel VMs); per machine with `cpp/tools/rt_validate.sh` |

**Latency measured in CI** ([run #17](https://github.com/Zautzik/Agrios/actions/runs/40004074946)): wake-up latency in µs, 60,000 samples per tool, on shared Azure VMs with a stock kernel under `stress-ng` load. These absolute values describe those VMs, not a real-time bound.

| p50 / p99 / p99.9 / max | `cyclictest` (platform baseline) | Agrios control loop | Page faults in the loop |
|---|---|---|---|
| x86-64 (Xeon Platinum 8573C) | 6 / 213 / 313 / 2,266 | 7 / 26 / 288 / 1,984 | 0 |
| arm64 | 4 / 20 / 77 / 1,355 | 4 / 18 / 44 / 1,524 | 0 |

- **Real-time safety of the control step: verified two independent ways.** The per-period work is one function, `control_tick()`, marked `[[clang::nonblocking]]`. Clang's function-effect analysis rejects, at compile time, any call from it to something not provably nonblocking, including inside the bridge's C API, which is annotated and checked the same way. RealtimeSanitizer covers the same function at run time and aborts on any `malloc`/`free`, mutex, or blocking call reached while it runs. Neither check is trusted on faith. Planting a single `printf` in `control_tick` breaks the build. Each check also has a negative control CI requires to fail. The first draft of the runtime one "passed" because the compiler had legally optimized its `new`/`delete` pair away, so nothing was allocated for the sanitizer to see. That's exactly the silent-pass failure mode the control exists to catch, caught on itself.
- **The lock-free algorithm: verified.** A real producer thread against a real consumer thread, 5,000,000 items through a deliberately small (256-slot) buffer forcing constant wraparound — zero lost, duplicated, or reordered items, clean under `-Wall -Wextra -Werror`.
- **The cross-process, cross-language bridge: verified live, not asserted from reading the code.** A real `motor_control_rt_loop` process running; a *separate* Python process importing the actual `spatial_tools` module and calling the actual `@tool`-decorated functions — `submit_spatial_intent` delivered a pose, confirmed independently by the RT loop's own telemetry log switching from `pose=no` to `pose=yes` at the exact iteration the push landed; `read_joint_states` then read a live sample back.
- **The failure path: verified, not just the happy path.** `SCHED_FIFO` and `mlockall` both need `CAP_SYS_NICE`/`CAP_IPC_LOCK` or root. Without them, the loop correctly refuses to start and reports why (`Operation not permitted`) instead of silently continuing at normal scheduling — tested by actually removing those capabilities and confirming the refusal, not assumed from the code.
- **Latency: measured against the standard benchmark, not in isolation.** A latency number means little without knowing what the platform itself delivers. `cpp/tools/rt_validate.sh` runs `cyclictest` and this loop on the same core, at the same priority and period, under the same `stress-ng` load, and writes a side-by-side report. To keep the comparison even:
  - the load warms up before anything is measured;
  - the runs are interleaved (cyclictest, loop, loop, cyclictest);
  - both tools' non-RT threads stay off the measured core;
  - every statistic for both is computed the same way, from merged histograms.

  The absolute numbers belong to the platform; what matters is whether the loop's distribution tracks the baseline's. The first CI run, before those rules, showed the loop's tail up to *ten times better* than `cyclictest`'s. That's impossible for a loop the kernel wakes the same way, so it pointed at a biased comparison. With the rules in place, the arm64 runner agrees within a few microseconds at p99. On the x86 runners, `cyclictest` still shows a heavier p99 tail than the loop in every run (about 1% of its wake-ups past 200 µs), even with interleaving and identical thread placement. The cause isn't pinned down, so it isn't claimed as a win for the loop. The report now includes each run separately, so run-to-run variation is visible next to any gap (details in [NOTES.md](NOTES.md)). CI runs it on every push, on both architectures, and publishes the table in the job summary. Those runners are shared cloud VMs on a stock kernel, so their absolute figures say nothing about a real-time bound. An absolute bound is a property of a specific machine (kernel, firmware, core isolation), so it's measured per machine with the same script on a PREEMPT_RT kernel. The earlier hand measurements inside Docker on WSL2 were never a real-time claim; this replaces them.
- **Page faults inside the RT loop: measured, zero, after fixing the one it found.** The RT thread records its own page-fault count immediately before and after the loop. The first measurement found exactly one minor fault per run: the loop's first `clock_gettime()`, which touched the lazily set-up vDSO mapping. A clock read during setup moved it out of the loop. It's now re-measured at zero, and CI fails if that ever changes. This also closes the question the `MAP_POPULATE` audit below left open (whether faults during actual hot-loop operation were zero) with a measurement that reaches the hot loop.
- **ThreadSanitizer: verified clean, in CI, on x86-64 and arm64.** The 5,000,000-item SPSC stress test, built with `-fsanitize=thread` and run with `halt_on_error=1`, passes on every pull request on GitHub-hosted `ubuntu-24.04` and `ubuntu-24.04-arm` runners ([first passing run](https://github.com/Zautzik/Agrios/actions/runs/39283625508)). The same job also runs `rt_safety_test` under TSan, which drives the shared-memory bridge (the `shm_open`/`mmap`-backed rings and the C API) and the control step from several threads. Scope, stated plainly: TSan sees one process, so a race *across* the process boundary is beyond it; the same code path within one process is covered. The earlier crash on launch inside Docker on WSL2 (`unexpected memory mapping`) matches a known TSan failure mode on kernels with high mmap ASLR entropy; CI guards against it by lowering `vm.mmap_rnd_bits` to 28 before running. Whether that was the cause on the WSL2 machine was not re-tested there.
- **Weakly-ordered hardware: verified for the algorithm, not yet for the whole bridge.** The same stress test, plain and under TSan, passes on native arm64 runners — the memory model where a missing acquire/release shows up as a stale or torn payload instead of being masked, as x86-64's stronger ordering would. The RT loop also runs on arm64 in CI: the `rt-safety` job drives the real `motor_control_rt_loop` binary for 5,000 periods at `SCHED_FIFO`, and `rt_safety_test` runs the control step over real shared-memory bridges with a live producer thread. The two-process path now runs on arm64 too: the `bridge-two-process` CI job starts the real RT loop, then a separate Python process uses the agent's own spatial tools against it. TSan can't see across processes, so the job checks the data instead. Every joint sample read across the boundary must satisfy the waveform's sine identity, which a torn read breaks (a negative control proves the check catches one). The loop's own count of poses received must also equal what Python delivered. Locally on x86-64: 11,909 samples, 0 torn, 4,592 poses sent and 4,592 counted. That's evidence from data, not proof: a race that never fired in these runs wouldn't show.
- **Teardown/signal handling: audited, two real bugs found and fixed.** Neither owner process (`motor_control_consumer`, `motor_control_rt_loop`) originally handled `SIGTERM` — one didn't handle any signal at all — meaning the default termination path skipped `shm_unlink` and orphaned segments in `/dev/shm`. This was live for the entire session and never surfaced, because every verification run used an ephemeral `docker run --rm` container, which destroys `/dev/shm` on exit regardless of whether cleanup code ran. Fixed and reverified directly: sent real `SIGTERM` to both binaries, confirmed each exited on its own (not force-killed) and left `/dev/shm` empty afterward.
- **ABI layout and first-touch page faults: audited against four specific vectors, two real, two not.** `#pragma pack`/extra reordering barriers were requested and correctly *not* added — the structs already have zero padding (proven by a `sizeof` static_assert passing ~10 real compilations) and `std::atomic` acquire/release already is the formal barrier; adding either would have been a regression, not a fix, and that's stated with the reasoning, not just asserted. A demanded conversion of the hot loop to a busy-spin was also declined: it doesn't spin (it blocks on `clock_nanosleep`), and spinning at 1 kHz would burn more power, not less. What *was* real: `mmap()` never used `MAP_POPULATE`, and the ring buffer's payload region was never touched at construction, only `head_`/`tail_` — fixed with an explicit prefault pass (write-touch for the owner, read-only touch for non-owners, since a resident physical page still needs each process's own page-table entry). Also strengthened the cross-language ABI check from a `sizeof`-only comparison to per-field `offsetof` checks, closing a real gap the size check couldn't catch (matching size, silently diverged field order).

Full verification log, including the real bugs the build process itself caught (a `static_assert(offsetof(...))` that didn't compile on an incomplete template type; a concurrency test whose `assert()`-based checks were silently no-ops under `NDEBUG`) — in [NOTES.md](NOTES.md) and [cpp/README.md](cpp/README.md).

## The agent driving it today

The pose intents flowing into the bridge above come from a LangGraph ReAct-style agent reasoning over natural language, on a local Ollama model (`qwen2.5:7b`) — not yet a perception pipeline. The agent itself is a small, fully-instrumented farm-operations assistant: SQL-backed persistence, full tracing, and a 30-case live-model eval suite, described below. It's the thing this repo actually has running end to end today; the bridge is what it's designed to hand off to once a real perception/policy producer is wired in.

```
                ┌──────────┐
   START ─────► │  agent   │ ◄────────┐
                └────┬─────┘          │
                     │  tool_calls?   │
              ┌──────┴──────┐         │
             yes            no        │
              ▼              ▼        │
         ┌─────────┐       END        │
         │  tools  │ ────────────────-┘
         └─────────┘
```

- **`agent` node** — calls the Ollama model (bound to 7 tools) with the running message history.
- **`tools` node** — executes whichever tool(s) the model requested, returns results as `ToolMessage`s.
- **Conditional edge** — routes back to `tools` only while the model keeps requesting them; exits to `END` the moment it answers directly.
- **Checkpointing** — graph state is persisted per `thread_id`, so multi-turn memory survives a process restart, not just a single run. It follows `DATABASE_URL`: by default `SqliteSaver` writes to `checkpoints.db`; with a Postgres URL (`DATABASE_URL=postgresql+psycopg://user:pass@host/db`), `PostgresSaver` stores conversations in the same database as the farm data. Switching doesn't copy existing SQLite history across.

### Demo

A real, unedited transcript from a live run against `qwen2.5:7b` — a plain conversational turn followed by a tool-calling turn resolved against the actual SQLite-backed field data:

```
You: Hi, who are you?
Hello! I'm Qwen, an AI assistant created by Alibaba Cloud. My main goal is to help you with
any questions or tasks related to agriculture, weather forecasts, and more. How can I assist
you today? Whether you need information about planting, managing fields, or anything else
related to your farming activities, feel free to ask!

You: What is the status of field-1?
The current status of field-1 is that it contains potatoes which are currently growing.

Would you like to log a task for field-1 or need more information about the crop?
```

(`query_field_status(field_id="field-1")` was called under the hood and returned the real row from `agrios.db`.)

### Tools

| Tool | Purpose |
|---|---|
| `get_weather_forecast(location, days=7)` | Forecast for planting/irrigation/frost-risk timing |
| `lookup_crop_calendar(crop, region)` | Planting and harvest windows for a crop |
| `query_field_status(field_id)` | Current crop and status for a field |
| `log_task(field_id, description)` | Records a pending task against a field |
| `get_pending_tasks()` | Lists all pending tasks across fields |
| `submit_spatial_intent(tvec, rvec)` | Pydantic-validated 6D pose, pushed onto the C++ motor-control bridge |
| `read_joint_states()` | Reads live joint-angle telemetry from the motor-control bridge |

`lookup_crop_calendar` calls FAO's [Crop Calendar](https://cropcalendar.apps.fao.org/) API for Chile (public, no key), and maps the region onto FAO's agro-ecological zones. `get_weather_forecast` calls [Open-Meteo](https://open-meteo.com/) (free, no API key): a `field_id` uses the field's stored coordinates, anything else is geocoded, each day carries a `frost_risk` flag (minimum ≤ 2 °C), the result lists `rain_days` (≥ 1 mm) and `frost_risk_days`, and a failed lookup returns an error rather than made-up numbers. `lookup_crop_calendar` still returns fixture data. The last two are defined in [spatial_tools.py](spatial_tools.py), not `main.py`, and degrade to a plain string result (not a crash) when the C++ side isn't available — true by default on this project's own Windows dev machine, since POSIX shared memory doesn't exist there.

### Design decisions

A few choices here were deliberate, not defaults — worth knowing why if you're reading the code:

- **Recursion limit set explicitly (`10`), not left at LangGraph's default (`25`).** The `agent → tools → agent` cycle's exit condition depends entirely on model judgment; a local 7B model can get stuck re-calling a tool with bad args. Capping it tighter than the library default catches a stuck loop faster, and it's caught explicitly via `GraphRecursionError` so the CLI degrades to a message instead of crashing.
- **Exception handling is scoped to where failure is actually possible, not blanket-applied.** The three tools that touch SQLite catch `SQLAlchemyError`, scoped to real I/O. The two tools that return static data don't have a try/except at all — there's nothing there that can fail yet, and a bare `except Exception` around code that can't throw just masks future bugs instead of guarding anything real. At the model-call boundary, the ollama client's streaming and non-streaming code paths have different exception contracts: `_request_raw()` wraps `httpx.ConnectError` in Python's builtin `ConnectionError`, but the streaming path used by `ChatOllama` (`inner()`) does not — it lets `httpx.ConnectError` propagate unwrapped. Catching the right types required reading the client source, not guessing from the class hierarchy. The except tuple covers `ollama.ResponseError` (model-side error), `ConnectionError` (non-streaming path, server unreachable), and `httpx.ConnectError` (streaming path, TCP connection refused) — each for a confirmed failure mode, not defensive coverage.
- **Domain data sits behind a SQLAlchemy engine, not raw `sqlite3` calls.** Every tool goes through one `engine` object with explicit connection pooling (`QueuePool`). At this scale that's not solving a real concurrency problem — it's a single-threaded CLI — but it means the only thing that changes to move to Postgres later is the connection string on one line; no tool function needs to change.
- **Database paths are anchored to the project directory, not CWD.** Both `agrios.db` and `checkpoints.db` resolve through `Path(__file__).parent` rather than relative URI strings. A relative path in a SQLAlchemy connection string (`sqlite:///file.db`) resolves to whatever the process's current working directory is at runtime — not where the code lives. That's an ambient property of how the script was invoked, not a stable code decision. Anchoring to `__file__` makes the file location an explicit invariant rather than something that silently varies depending on where you launched the process from.
- **Tracing via self-hosted Langfuse, not LangSmith.** A `CallbackHandler` is wired into the LangGraph `invoke` config, giving full per-node trace spans (model input/output, tool name/args/result) without depending on a paid hosted service.

### Evaluation

A 33-case parametrized suite (`test_agent.py`, run via `pytest -m slow`) checks the live model
across five categories: tool selection (15 cases), argument correctness (8), multi-step task
completion (5), no-tool-needed (2), and answer faithfulness (3). The faithfulness cases pin the
forecast to known data and check that the final answer agrees with it, which the other
categories can't: they only check which tools were called. It's gated behind a marker so a plain `pytest` run stays
fast and never touches Ollama; the full suite takes ~25 minutes on CPU against `qwen2.5:7b`.

**Latest result:** 33/33: 32 in one full run (45m 30s on CPU, slowed by Docker holding RAM), and the dry-forecast case 5/5 on its own after fixing two bugs in that test. Its first version used 0.2 mm on one day and then failed the model for faithfully reporting 0.2 mm of rain; its first matcher rejected the correct "there is no expected rain". Earlier, the 30-case suite scored 30/30 three times in a row,
after the `region` column, the tool-call guard, and live weather; before those, 29/30.
- **Weather gap: closed.** *"What's the weather forecast for field-1?"* used to fail 3/3. With
  the docstring saying `location` can be a `field_id` and that every weather question must call
  the tool, it passed 3/3 in isolated runs and again in the full run.
- **Fallow-field case: fixed.** *"What's in field-2 and when should I harvest per the crop
  calendar"* used to call `lookup_crop_calendar` only about half the time (4–5 of 8 isolated
  runs), because field-2 is fallow and the model decided there was nothing to look up. A
  stronger docstring didn't help (3/5). What did: `query_field_status` now adds a note to a
  fallow field's result saying the crop is the planned crop and pointing at the calendar.
  Isolated: 8/8, every call with `wheat`/`central Chile`.
- **Latest full run, after real weather and summaries:** 30/30 (31m 41s; slower than earlier runs because Docker was holding about 4.5 GB of RAM). The weather cases now call the real Open-Meteo, so they also need network access.
- **Invented arguments: fixed in two steps.** Fields had no region, so the model made one up
  ("Midwest USA", "generic-region"). With a `region` column that `query_field_status` returns,
  4/5 runs on the field-1 question passed the real `potato`/`central Chile` (0/5 before). The
  miss came from the model calling both tools *in the same turn*, before it had the field's
  data, and guessing. A guard in the tool step now refuses a calendar call made alongside
  `query_field_status`, and any call with an empty required argument, replying with why so the
  model can retry. With the guard: 6/6 runs ended on the correct call, including 2 where it
  caught a same-turn guess and the model retried with the real values. The guard can't catch a
  confident wrong value in a call made on its own (e.g. `"unknown"` as a region).
- **Tried and dropped:** telling the model in the calendar docstring to take the region from
  `query_field_status`. It stopped guessing, but asked the user instead of looking the field up:
  *"When can I harvest the potatoes I'm growing?"* fell to 1/5 tool calls.
- **Recursion-limit flake on the frost question:** didn't recur in this run. Still too little
  data to call it resolved.

The earlier 28/30 (5 tools, before these fixes) and the investigation behind it, including a
cold-start hypothesis that got tested and killed, are in [NOTES.md](NOTES.md).

## Perception harness (`vision/`)

A CCTV perception cell is being built in phases. Phase 1, done, is the evaluation harness, built before any model exists:
- MOTA and IDF1, cross-checked against py-motmetrics to 1e-9;
- False Alarms per Camera-Day with an exact Poisson bound;
- Gait Asymmetry Index error, and latency percentiles carrying a machine stamp;
- seeded synthetic scenes, camera-day track streams and quadruped gaits;
- a CI gate whose negative controls (noise input, shuffled time order) must fail every suite in it.

Metrics, measured results and design decisions are in [vision/README.md](vision/README.md).

| Gate suite (reference pipeline, synthetic) | Intact | Noise | Shuffled |
|---|---|---|---|
| Kinematics: FACD, one-sided 95% upper (3 camera-days) | 0.9986 | 347.1 | 369.3 |
| Scene: surge recall / MOTA | 0.75 / 0.496 | 0 / 0 | 0.5 / −0.203 |
| Gait: max GAI error (points) | 1.16 | 23.6 | 4.95 |

## Web console (`agrios/console/`, in progress)

A local web app over the same services the chat agent calls, planned in [docs/console-plan.md](docs/console-plan.md). It covers the farm features now and perception later. Built so far:
- first-run owner setup;
- sign-in with argon2id and server-side sessions;
- a Today page;
- Settings generated from typed models, with validation, defaults, reset and a change log.

To see every screen without a farm, use the demo, which runs on its own database:

```
uv sync --extra console
uv run python -m agrios.console --demo
```

Then open http://localhost:8000 and sign in with the demo account shown on the page. By default it's reachable from this machine only; **Settings > Network & access** changes that. The security choices and their references are in the plan's C1b section.

## Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent control flow as an explicit graph
- [LangChain](https://github.com/langchain-ai/langchain) + [`langchain-ollama`](https://github.com/langchain-ai/langchain) — model/tool binding
- [Ollama](https://ollama.com/) (`qwen2.5:7b`) — local inference, no API cost or external dependency
- [SQLAlchemy](https://www.sqlalchemy.org/) — pooled connections over SQLite (`agrios.db`) or Postgres, tested against both in CI
- `langgraph-checkpoint-sqlite` / `langgraph-checkpoint-postgres` — conversation persistence (`checkpoints.db`, or Postgres)
- [Langfuse](https://langfuse.com/) (self-hosted) — full tracing/observability
- [`uv`](https://github.com/astral-sh/uv) — Python dependency management
- C++20, CMake, POSIX shared memory, `pthread` — the real-time bridge (`cpp/`), Linux/macOS only
- Clang 20 function-effect analysis, LLVM RealtimeSanitizer, ThreadSanitizer, `cyclictest` + `stress-ng` — real-time verification, in CI on x86-64 and arm64
- NumPy, SciPy, OpenCV (headless); py-motmetrics as the reference implementation; ruff + `mypy --strict` — the perception harness (`vision/`), installed with `uv sync --all-extras`

Full build log → [NOTES.md](NOTES.md) &nbsp;·&nbsp; From-first-principles study guide → [docs/Masterclass-Agrios.md](docs/Masterclass-Agrios.md)

## Setup

**Prerequisites:** [Ollama](https://ollama.com/download) installed and running, [`uv`](https://github.com/astral-sh/uv) installed, [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for tracing, and for building/running `cpp/` on Windows — see [cpp/README.md](cpp/README.md)).

```bash
# 1. Pull the model
ollama pull qwen2.5:7b

# 2. Install dependencies
uv sync

# 3. (optional) bring up self-hosted Langfuse for tracing
git clone https://github.com/langfuse/langfuse.git langfuse-server
cd langfuse-server && docker compose up -d
# open http://localhost:3000, sign up, create a project, generate an API key
```

Create a `.env` in the project root:
```
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_BASE_URL="http://localhost:3000"
```

Run it:
```bash
uv run main.py
```
Type a message at the `You:` prompt; type `exit` to quit. Tables and seed data (`agrios.db`) are created automatically on first run.

To build and run the C++ bridge itself (Linux/macOS, or a container — see [cpp/README.md](cpp/README.md) for the full walkthrough including the real-time loop):
```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
```

## Known limitations

- **No VLA/perception pipeline is wired in yet.** `submit_spatial_intent` is exercised today by an LLM tool call reasoning from natural language, not by a vision or learned-policy model. The bridge is producer-agnostic by design, but nothing in this repo produces `tvec`/`rvec` from vision yet. `vision/` is at Phase 1: the harness that the perception models will be measured with, not the models.
- **No real hardware, no real joints.** `JointState6` telemetry is a synthetic waveform published by the RT loop, a stand-in for real encoder feedback — there is no motor, encoder, or physical arm behind this yet.
- **Latency figures so far come from stock-kernel CI VMs.** Real-time *safety* (no allocation, locks, blocking, or page faults in the loop) is platform-independent and verified above. An absolute worst-case latency is not: it belongs to a specific machine's kernel, firmware, and core isolation, so it's measured per machine with `cpp/tools/rt_validate.sh` on a PREEMPT_RT kernel. [cpp/README.md](cpp/README.md#measuring-latency-on-real-hardware) covers running it from a live USB without installing anything.
- Both data tools need network access: weather from Open-Meteo, crop calendars from FAO. Without it they return an error. The FAO calendar covers Chile only, by agro-ecological zone (central Chile is FAO's Mediterranean zone); any other region gets an error rather than made-up dates.
- One conversation per `--thread` id (default `"1"`), chosen on the command line — not per-user session handling. Only the last 20 messages, starting at a user turn, are sent to the model; the full history stays in `checkpoints.db`. That fixes the slowdown a long-lived thread caused (25+ messages replayed every turn, 30s–2min on CPU), and once a thread passes 20 messages, everything but the last ~10 is folded into a running summary that the agent sees as a system message. Tasks logged are read from the `log_task` results by code and kept as their own line, so the summary can't drop or invent one. That costs one extra model call each time it happens, and a summary is lossy: details it leaves out are gone from the model's view (the full messages stay in older checkpoints). To move existing history onto Postgres, run `migrate_checkpoints.py` with `DATABASE_URL` set; it copies every thread and is safe to re-run.
- `qwen2.5:7b` needs ~4.3 GB free RAM to load; on a constrained machine running Docker simultaneously, this can fail — handled gracefully (see Design decisions) rather than crashing, but worth knowing going in.
- Tracing needs the self-hosted Langfuse stack running in Docker. With it up, full agent traces export and are stored (checked against server 3.200.0 and SDK 4.12.0: agent step, `ChatOllama` generation with all 7 tools bound, routing). With Docker down, the agent still runs, but the background exporter logs `Failed to export span batch` errors. An earlier `404 Not Found` from the OTLP endpoint doesn't reproduce on the current stack, so its cause is unknown.
