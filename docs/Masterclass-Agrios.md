# Agrios — Masterclass & Study Guide
> A from-first-principles study guide to everything living in this repo: a ~170-line LangGraph
> agent that turned out to contain a full syllabus, plus a second, entirely separate subsystem
> that grew alongside it — a C++20 lock-free ring buffer over POSIX shared memory, hardened to
> PREEMPT_RT/ROS 2 real-time discipline, bridging that same agent to a 1 kHz motor-control
> thread. Two languages, two memory models, two failure domains, one repo. Explicit state
> machines, functional state reducers, durable checkpointing, tool-calling as structured
> generation, bounded recursion, failure-boundary discipline, connection pooling, observability,
> hermetic testing of a non-deterministic system — and, on the systems side, lock-free
> concurrency with formally correct memory ordering, cache-line-aware data layout, real-time
> thread scheduling, POSIX IPC lifecycle management, cross-language ABI design, and
> async-signal-safe process teardown. Every idea below is traceable to a specific line in
> `main.py`, `test_agent.py`, `cpp/`, `spatial_tools.py`, or a specific commit — nothing here is
> invented, and nowhere does this document claim a measured result the project doesn't actually
> have. The eval suite has been run for real: 28/30 on reconciliation, one confirmed reproducible
> gap, one open question — see war story #7 in Part 3. The latest run, with all seven tools and
> the weather fix, the `region` column and the tool-call guard, scored 30/30 twice (see Part 4). The bridge has been verified for real
> too, including two rounds of adversarial audit that each found genuine bugs and each correctly
> refused at least one demanded "fix" that would have been a regression — see war stories #13
> and #14, and Part 11 in full.
>
> Companion to [NOTES.md](../NOTES.md) (the build log, in the moment) and [README.md](../README.md)
> (the reference). This document is the third angle: not what happened or what it does, but
> *why the ideas underneath it work*, explained as if teaching them to someone who's never
> seen an agent framework — or a lock-free queue — before.
---
## Part 0 — The one-sentence summary
> *"Agrios is a LangGraph ReAct-style agent for farm operations, running entirely on a local
> Ollama model: seven tools, a hand-written state graph instead of a black-box `.run()` call,
> SQL-backed persistence that survives a restart, full tracing, and exception handling scoped
> to exactly the failures I've confirmed can happen — including two I only found by running the
> thing and reading a library's source code. Two of those seven tools cross into a second
> system entirely: a lock-free C++20 ring buffer over POSIX shared memory, hardened to
> PREEMPT_RT/ROS 2 discipline, that the Python agent talks to but never runs inside of."*
The project is deliberately small on the Python side, and deliberately narrow (one data
structure, reused twice, plus the scaffolding to run it safely) on the C++ side. Neither is a
limitation to apologize for — it's what makes every concept in both halves inspectable in full,
instead of buried under abstraction layers you have to take on faith. A production agent
framework hides the graph, the reducer, the checkpoint format, and the tool-calling wire
protocol behind convenience functions; a production robotics IPC layer usually hides the same
way, behind a message-passing library (ROS 2's DDS, ZeroMQ, gRPC) whose lock-free internals you
never read. This project wrote all of it out by hand, once, on purpose, specifically so each
piece could be understood rather than assumed.
---
## Part 1 — Skills inventory
Depth is marked honestly: **solid** (could explain and defend it under direct questioning) ·
**developing** (understand it, want more reps).
### 1A. Technical / hard skills
| Skill | What I can now do | Depth |
|---|---|---|
| **Agent control flow as an explicit graph** | Model an agent loop as nodes + edges (`StateGraph`) instead of a hidden `.run()` — explain why that buys checkpointing, tracing, and inspectability that a black-box loop can't | solid |
| **State & reducers** | Explain why `Annotated[list, add_messages]` is required for multi-turn memory to work at all, and what silently breaks without it | solid |
| **Conditional routing** | Write a routing function (`should_continue`) that inspects the last message and returns a node name or `END` — the mechanism every agent loop's exit condition reduces to | solid |
| **Durable checkpointing** | Swap `MemorySaver` → `SqliteSaver` as a storage-layer change behind one interface; explain what a checkpoint actually stores and why `thread_id` is the partition key | solid |
| **Tool / function calling** | Explain the path from a Python function + docstring to a JSON schema to a model's structured `tool_calls` output to an executed result fed back as a `ToolMessage` | solid |
| **Bounded recursion** | Set an explicit `recursion_limit`, explain why an agent/tools cycle has no built-in termination guarantee, and handle `GraphRecursionError` as a real, expected outcome | solid |
| **Narrow exception handling** | Scope every `except` to a boundary where a *specific, confirmed* failure can occur; delete a `try/except` that guards nothing rather than widen it | solid |
| **Reading library source over guessing** | Traced a bug through `ollama`'s Python client to two different exception contracts on two different code paths (streaming vs. non-streaming) — found by reading, not by pattern-matching a traceback | solid |
| **SQL via SQLAlchemy** | Move from raw `sqlite3` calls to a pooled `engine` + `text()`, understand what `QueuePool` actually does, and why the real payoff is a one-line connection-string swap, not the pooling itself at this scale | solid |
| **Environment vs. code invariants** | Diagnose a bug caused by a CWD-relative path resolving differently depending on where a process was launched from; fix it by anchoring to `Path(__file__).parent` | solid |
| **Observability** | Wire a `CallbackHandler` into a graph invocation for full per-node tracing (self-hosted Langfuse), distinct from and complementary to stdlib `logging` | solid |
| **Local LLM inference** | Run tool-calling inference entirely offline via Ollama; understand the real RAM ceiling this imposes, having hit it | solid |
| **Testing a non-deterministic system** | Design a parametrized eval suite across failure-mode categories, keep it hermetic against a system whose whole point is persistent shared state, and gate a slow/live suite behind a marker | solid |
| **Python craft in service of the above** | `TypedDict` + `Annotated` for typed, reducer-aware state; decorators (`@tool`); context managers (`with engine.connect()`, `with SqliteSaver.from_conn_string(...)`); dict-merge semantics; f-strings; `if __name__ == "__main__"` | solid |
| **Git workflow** | Coherent, single-purpose commits, each with a rationale in the message — not "fix stuff" | solid |
| **Lock-free concurrent data structures** | Design and implement a single-producer/single-consumer ring buffer from scratch — index math via bitwise AND on a power-of-two capacity, the full-vs-empty ambiguity resolved by always keeping one slot unused rather than a separate counter | solid |
| **C++ memory-order semantics** | Choose `relaxed`/`acquire`/`release` per operation, not `seq_cst` by default — explain exactly which store-load pair synchronizes-with which, and why a plain (non-atomic) payload write is safely ordered by the atomic release/acquire around it, not by being atomic itself | solid |
| **Cache-line-aware data layout** | Diagnose false sharing as a *correctness-adjacent performance* bug (not a crash, a stall) and fix it with `alignas(64)` on independently-written fields; verify the resulting layout at compile time instead of trusting it | solid |
| **POSIX shared memory / cross-process IPC** | `shm_open`/`mmap`/`ftruncate`/`munmap`/`shm_unlink` end to end — owner-vs-attacher lifecycle, stale-segment cleanup, and the distinction between a mapping (per-process, kernel-reclaimed automatically) and a named object (persists until explicitly unlinked) | solid |
| **Real-time thread scheduling (PREEMPT_RT/ROS 2 style)** | `mlockall` + explicit stack pre-fault, `SCHED_FIFO` priority selection (and its kernel-thread-starvation risk at 99), CPU affinity vs. true core isolation, absolute-deadline `clock_nanosleep` scheduling vs. relative-sleep drift | solid |
| **Cross-language ABI design** | Build a compiled C boundary between Python and C++ instead of letting `ctypes` reimplement synchronization logic; verify layout compatibility with per-field `offsetof`, not just `sizeof` | solid |
| **Signal handling & process lifecycle** | Distinguish `SIGINT` from `SIGTERM` and know why a real supervisor sends the latter; know exactly what a signal handler is allowed to touch (`std::atomic`, checked via `is_always_lock_free`) and why almost everything else in the standard library isn't | solid |
| **Concurrency correctness verification** | Design a stress test that forces the failure mode it's checking for (a small buffer, forced wraparound, millions of operations) rather than hoping a single pass proves anything; caught my own test silently checking nothing under `NDEBUG` before trusting a green result again | solid |
| **Compile-time layout verification (C++ templates)** | Use `static_assert(offsetof(...))` to make an ABI assumption fail the build instead of failing silently in production; diagnose and fix an "incomplete type" error by understanding what a "complete-class context" actually is | solid |
| **Cross-platform build systems** | `CMake` targets with per-target compiler/linker flags (`-Wl,-z,now`), a platform guard that refuses to configure on the wrong OS on purpose, and containerized (Docker) verification for POSIX code on a Windows dev machine | solid |
### 1B. Cognitive habits
These outlast this specific stack. Anyone can pick up `SqliteSaver` in an afternoon; these are harder to learn.
| Habit | The move | Where I did it |
|---|---|---|
| **Read the source, not the stack trace** | When a caught-exception assumption turned out wrong, went into the `ollama` client's actual code instead of widening the `except` | The `httpx.ConnectError` bug — two code paths, two different exception contracts, found at specific line numbers |
| **Isolate & inspect** | When output didn't match the code I was reading, stopped trusting the editor tab | The stale-interpreter `NameError` — the fix was already saved to disk, the running process wasn't the saved file |
| **Make inherited defaults into decisions** | Noticed I was relying on LangGraph's default recursion limit (25) without ever having chosen it | Set it explicitly to 10, with a reason (a 7B model can misread a tool result and re-call it) |
| **Delete code that isn't guarding anything** | Two tools had a `try/except Exception` around a hardcoded string return — no I/O, nothing that could fail | Removed it instead of narrowing it; a wider or narrower `except` around a function that can't throw is still theater |
| **Don't build for a step you haven't taken yet** | The SQLAlchemy engine made a later Postgres swap a configuration change — and stopped there. When it was finally tested, the one blocker was dialect-specific DDL (`AUTOINCREMENT`), fixed by moving to SQLAlchemy Core tables; no Postgres-specific code was needed | NOTES.md kept it as an open question until a real Postgres run and a CI job backed it |
| **Test hermetically, not against shared state** | Recognized that the app's own persistence (`SqliteSaver` + fixed `thread_id`) would make eval cases leak into each other if reused as-is | `MemorySaver` + a fresh `uuid4()` thread per test case |
| **Name the environment as a bug source, not just the code** | An OOM crash and a CWD-relative path bug were both "correct code, wrong assumption about what's running underneath it" | The Docker/WSL memory-pressure incident; the subdirectory-launch path bug |
| **Predict-then-verify** | State a plausible explanation for a failure, then test it directly instead of writing it down as fact | The cold-start theory for the weather-forecast miss — killed by rerunning the case alone twice more instead of trusting the first guess |
| **Decline a wrong instruction, with reasoning, not silent compliance** | Two separate audits demanded specific "fixes"; two of the four demanded changes were based on incorrect premises about the code and would have been regressions | Refused `#pragma pack(1)` (would risk unaligned access, there's no padding to remove) and a busy-spin conversion (would burn *more* power at 1 kHz, not less) — explained why in both cases instead of complying because the request was phrased forcefully |
| **Distinguish "looks alarming" from "is actually a bug"** | An audit's framing assumed the Python side should call `shm_unlink` on shutdown | It shouldn't and doesn't — it's a non-owner by design, and "fixing" that would delete a segment a still-running producer depends on; verified the *actual* concern (missing `munmap`) was already covered by kernel-guaranteed cleanup, not left unexamined |
| **Report a measurement that contradicts the hypothesis, don't reframe it** | A before/after page-fault comparison came back showing *more* faults in the patched version, not fewer | Wrote down the number as measured, then explained precisely why it was still consistent with the fix working (the test never reached the code path the fix targets) — instead of quietly picking a different metric that told a cleaner story |
| **An ephemeral test environment can hide a real bug indefinitely** | Two real signal-handling bugs (one binary with zero handling, one missing `SIGTERM` specifically) were live through dozens of verification runs and never surfaced | Every run used `docker run --rm`, which destroys `/dev/shm` on exit regardless of whether cleanup code ran — a graceful exit and an orphaned segment looked identical from outside the container every single time |
---
## Part 2 — Where the project stands
Every phase below is one real commit — this is the actual build order, not a retrofit narrative.
| Phase | Commit | What changed | The concept it introduced |
|---|---|---|---|
| 0 — Scaffold | `eaf0590` | `create_react_agent` + one tool (`get_weather_forecast`), a single `.invoke()`, no memory | The prebuilt agent constructor: convenient, opaque |
| 1 — Memory | `e6c266d` | Swapped to `langchain.agents.create_agent`, added `MemorySaver` + `thread_id`, replaced the one-shot invoke with an `input()` loop | Checkpointing exists, but only for the life of the process |
| 2 — The real graph | `2b255e1` | Hand-rolled `StateGraph` (`agent`/`tools` nodes, conditional edge, `Annotated[list, add_messages]`), grew to 5 tools, `SqliteSaver` for durable persistence, `try/except` on every tool (later found to be over-applied), structured logging, Langfuse `CallbackHandler` | Control flow becomes an inspectable graph instead of a framework's hidden loop |
| 3 — Hardening pass 1 | `fa4350b` | Explicit `recursion_limit=10` + `GraphRecursionError` handling, deleted the dead `try/except` on 2 static tools, moved the 3 DB tools onto a pooled SQLAlchemy `engine`, caught `ollama.ResponseError`/`ConnectionError` after a real out-of-memory crash | Exception handling narrows to real boundaries instead of blanket coverage |
| 4 — Hardening pass 2 + eval | `1edd0c8` | 30-case parametrized `pytest` eval suite (`test_agent.py`) with hermetic `MemorySaver` + `uuid4` runs behind a `slow` marker; fixed `httpx.ConnectError` bypassing the existing except tuple on the streaming code path; anchored both DB paths to `Path(__file__).parent`; removed an unused leftover import | Verification becomes repeatable instead of manual, and a real transport-layer bug gets found and fixed |
| 5 — Eval run + docs | `70d5de8` | Ran the 30-case suite live against `qwen2.5:7b`, fixed one test-bug (potato/potatoes exact match), reconciled to 28/30 | A passing/failing test result is a claim, not a fact, until you've read *why* |
| 6 — The bridge, part one | `1b4522d` | `cpp/`: `SPSCRingBuffer<T,Capacity>` (lock-free, `alignas(64)`, acquire/release), `SharedRingBuffer` (POSIX `shm_open`/`mmap` wrapper), a compiled C ABI (`bridge_c_api.cpp`) instead of raw Python `ctypes` struct-poking, a 5M-item concurrency stress test | Lock-free concurrent data structures and cross-process, cross-language IPC enter the codebase for the first time |
| 7 — The real-time loop | `b425d12` | `motor_control_rt_loop.cpp` + `rt_thread.hpp`: `mlockall` + stack pre-fault, `SCHED_FIFO` 99, CPU affinity, absolute-deadline `clock_nanosleep` scheduling, zero-allocation hot loop with telemetry exported through a second lock-free channel | PREEMPT_RT/ROS 2 real-time thread discipline, verified both when it succeeds and when it correctly refuses to run |
| 8 — LangGraph ↔ bridge integration | `863e5fd` | `spatial_tools.py`: `SubmitSpatialIntent` (Pydantic `args_schema`), `read_joint_states`; migrated `Pose6D` from placeholder Euler angles to `tvec`/`rvec` (a breaking ABI change, confirmed necessary before making it); added a second, independent shared-memory channel for joint telemetry | The Python agent and the C++ real-time loop become one system, verified end to end through a real `@tool` call into a real running RT process |
| 9 — Two adversarial audits | `ac24fd0`, `96f5c6a` | Fixed a real `SIGTERM`-handling gap in both owner processes (bug #1: zero signal handling; bug #2: `SIGINT` only) that had been silently masked by every prior `docker run --rm` test; fixed a real `mmap()` prefault gap (`MAP_POPULATE` + explicit touch) and strengthened an ABI check from `sizeof`-only to per-field `offsetof`; correctly declined two other demanded "fixes" (`#pragma pack`, a busy-spin conversion) as regressions | Auditing your own system adversarially finds real bugs that verifying it optimistically doesn't — and refusing a wrong fix is as much the skill as finding a right one |
**Current state, stated plainly:** two systems in one repo. The Python side: a single-file
CLI agent, seven tools (three DB-backed, two live APIs, two crossing into the C++
bridge), one hardcoded `thread_id`, local-only inference, traced through a self-hosted Langfuse
stack, checked by a 33-case suite (33/33: 32 in one full run (45m 30s on CPU, slowed by Docker holding RAM), and the dry-forecast case 5/5 on its own after fixing two bugs in that test. Its first version used 0.2 mm on one day and then failed the model for faithfully reporting 0.2 mm of rain; its first matcher rejected the correct "there is no expected rain") (28/30 before the
weather fix — see Part 4). The C++ side: a lock-free SPSC ring buffer proven
correct under a 5,000,000-operation concurrent stress test, two independent shared-memory
channels, a real-time thread hardened to PREEMPT_RT/ROS 2 discipline and verified live against a
real Python tool call — with an explicit, written-down account of exactly which parts of that
verification depend on real PREEMPT_RT hardware to mean anything, and which don't. See war story
#7 in Part 3, Part 4 for the honest roadmap, and Part 11 for the bridge's full technical
walkthrough.
---
## Part 3 — War stories
1. **The stale interpreter.** A `NameError` kept firing against code that, on inspection, no
   longer had the bug. The file on disk was already fixed — the process actually executing was
   a stale one from an earlier run. *Lesson: when behavior doesn't match the code you're
   reading, check what's actually running before you re-check the code.*
2. **The exception-contract archaeology.** An `except (ResponseError, ConnectionError)` block
   looked complete — model-side errors and unreachable-server errors, both covered. Then a
   real connection-refused crash slipped through both. Reading the `ollama` client source
   (not guessing from the traceback) turned up two different code paths with two different
   exception contracts: `_request_raw()` catches `httpx.ConnectError` and re-raises it as a
   builtin `ConnectionError`; but `ChatOllama` uses the *streaming* path, whose `inner()`
   function opens `self._client.stream()` in a `with` block — if the TCP handshake fails at
   `__enter__`, `httpx.ConnectError` propagates raw, before any local `except` block in that
   function even runs. Same library, two paths, two contracts. Fix: add `httpx.ConnectError`
   to the tuple directly. *Lesson: a class hierarchy you haven't verified is a guess with good
   posture — the source is the only ground truth.*
3. **The real crash that exposed a real gap.** Hit
   `ollama._types.ResponseError: model requires more system memory (4.3 GiB) than is available
   (4.1 GiB)` mid-session. Checked actual memory instead of assuming: ~1 GB free out of 15.7 GB,
   with `vmmemWSL` — running the *entire* self-hosted Langfuse stack (Postgres, ClickHouse,
   Redis, MinIO, two Langfuse services) — holding 2.4+ GB continuously in the background for a
   tool used interactively maybe once an hour. Two separate fixes followed: an environment fix
   (stop the containers when not actively tracing) and a code fix the crash exposed (the invoke
   call only caught `GraphRecursionError`, so *any* other failure took the whole REPL down —
   added `ResponseError`/`ConnectionError` handling, scoped to what the `ollama` client
   actually raises). *Lesson: a resource crash is real signal about your environment, not just
   an exception to swallow.*
4. **The CWD-relative path bug.** Running the agent from inside the `langfuse-server/`
   subdirectory silently created brand-new, empty `agrios.db` and `checkpoints.db` files right
   there — no error, no warning, just a cold-start agent with no memory and an empty task
   table. Cause: `"sqlite:///agrios.db"` is a *process-CWD-relative* URI in SQLAlchemy (three
   slashes = relative path), so a string that reads like a fixed file path is actually resolved
   at runtime against wherever the shell happened to be when the process launched. Fixed by
   anchoring both connection strings to `Path(__file__).parent` — the directory `main.py`
   itself lives in, which cannot change no matter where the process is invoked from. *Lesson:
   anything that needs to persist across runs needs a path that's an explicit invariant in the
   code, not something inherited from the shell.*
5. **The try/except that guarded nothing.** Two of the five tools (`get_weather_forecast`,
   `lookup_crop_calendar`) had a `try/except Exception` wrapped around a hardcoded string
   return — no I/O, no real failure mode, nothing that block could ever actually catch. It had
   been added for consistency with the three DB tools, without asking what it was protecting
   against. The fix wasn't a narrower except clause — it was deleting the block entirely; a
   different-but-still-fake except would just be a different way of pretending there's a real
   failure mode there. The three DB tools kept their scoped `except SQLAlchemyError`, because
   those genuinely can fail. *Lesson: a try/except should mark a real boundary where failure is
   possible, not get copy-pasted onto every function out of habit.*
6. **Eval hermeticity.** The live app deliberately reuses `SqliteSaver` and a fixed
   `thread_id="1")` — that's what makes multi-turn memory durable. Reusing that exact setup for
   automated tests would have made test case N's outcome depend on whatever case N−1 left
   behind, since `SqliteSaver` persists by `thread_id` across invocations. The fix: tests
   compile the graph with an in-memory `MemorySaver()` once per session, and every call mints a
   fresh `uuid4()` thread — thirty isolated conversations sharing one compiled app object, none
   able to see another's history. *Lesson: the same feature that makes a system useful
   (persistent shared state) is exactly what makes testing it dangerous by default — hermetic
   tests have to actively opt out of the thing the product is for.*
7. **The eval suite's first real run, and a hypothesis that died on contact with a second test.**
   Ran the full 30-case suite for the first time: 27/30 on the first pass, 24m33s on CPU. One
   failure (`crop="potato"` vs. the model's `"potatoes"`) turned out to be the test's exact-match
   assertion being stricter than it needed to be — fixed by loosening it to the substring check
   already used elsewhere in the same file, the same "inspect before you blame the model" move as
   war story #2 and Agroteca's own q18. One failure (a recursion-limit hit on a frost-risk
   question) didn't reproduce on retry — one pass after one fail, not enough samples to call it
   either resolved or real, so it's a watch item, not a fix. The last one is the one worth
   remembering: `"What's the weather forecast for field-1?"` — wording that echoes the tool's own
   docstring almost verbatim — failed to trigger `get_weather_forecast` at all, while every
   paraphrase of the same request succeeded. First guess was a cold-start effect (it's the very
   first live model call of the session); tested that guess directly by rerunning the identical
   case alone, twice more, in fresh processes. Failed both times — **3 for 3**. Not a session
   artifact; a specific, reproducible phrasing gap. *Lesson: a hypothesis is worth having, but
   only worth keeping if it survives being tested on purpose — the same discipline Agroteca's
   embedder experiment ran on itself, just applied here in about two minutes instead of seven
   hours.*
8. **The layout check that didn't compile.** `spsc_ring_buffer.hpp` asserts at compile time that
   `head_`, `tail_`, and the payload buffer land at the exact byte offsets the C ABI and the
   Python `ctypes.Structure` both depend on —
   `static_assert(offsetof(SPSCRingBuffer, head_) == 0)`, written directly in the class body.
   First build: `invalid use of incomplete type`. `offsetof` needs a complete type, and a class
   template isn't complete yet at the point in its own body where that assertion was written —
   the closing `};` hasn't been reached. Fixed by moving the `offsetof` calls into small
   `static constexpr` member *functions* — function bodies are a "complete-class context" even
   when written inline, so `offsetof` works there — then calling those functions from
   `static_assert`s placed after the class closes. *Lesson: a check that doesn't compile is a
   check that was never actually run — this one failed loudly at build time instead of silently
   passing nothing, which is the entire reason to write it as a `static_assert` instead of a
   comment.*
9. **The test that passed by checking nothing.** The concurrency stress test (a real producer
   thread against a real consumer thread, 5,000,000 items through a deliberately small buffer)
   was built on `assert()`. `cmake --build`'s default build type (`RelWithDebInfo`, like every
   CMake Release-family type) defines `NDEBUG`, which compiles every `assert()` in the program
   down to nothing. `ctest` reported "1/1 tests passed" — truthfully, about a test now checking
   zero of its conditions. The tell wasn't in the test output, it was in the build log: a warning
   about an unused variable that was only ever *read* inside `assert()` calls — once those
   calls compile away, so does the only place that variable does anything. Fixed by replacing
   every `assert()` with a small `CHECK()` macro carrying no `NDEBUG` guard at all, then verified
   the fix two independent ways (an explicit `-DNDEBUG` build, and a fresh CMake build) rather
   than trusting that adding the macro was self-evidently correct. *Lesson: the exact same shape
   of bug as the potato/potatoes test assertion from war story #7 — a green result is a claim
   about the test, not a fact about the code, until you've checked what the test was actually
   capable of catching.*
10. **A request that would have destroyed a working demo if honored too literally.** Asked to
   "refactor the 1 kHz motor control loop" to add real-time hardening. But the existing
   `motor_control_consumer.cpp` was never a 1 kHz loop — it's a poll-with-sleep-and-print demo
   built specifically to verify the bridge itself, and PREEMPT_RT constraints (no `std::cout`,
   no bounded exit, fixed-rate forever) would have broken that demo's actual job to satisfy a
   request that was really describing something that didn't exist yet. Wrote the real-time loop
   as its own new file instead, reusing the same bridge underneath, and said so explicitly rather
   than silently picking one interpretation. *Lesson: honoring an instruction's literal wording
   over its actual intent can be its own kind of failure to listen — the fix wasn't refusing the
   ask or silently overriding it, it was naming the mismatch out loud and then doing the thing
   that was actually being asked for.*
11. **A breaking ABI change, confirmed before it was made, not after.** A new tool needed to
   accept `tvec`/`rvec` (translation + Rodrigues rotation vector — what `cv2.solvePnP`/ArUco/
   FoundationPose emit natively). The existing `Pose6D` struct stored Euler angles instead — a
   placeholder chosen before any real producer existed to match, not a considered decision.
   Converting `rvec` to Euler *inside the tool* would have buried a lossy, gimbal-lock-prone
   transform behind what should have been a "write bytes to shared memory" function. The
   alternative — migrating `Pose6D` itself — meant touching a struct whose exact byte offsets are
   asserted at compile time and mirrored across three languages (C++, the C ABI, Python
   `ctypes`), a change with real blast radius. Confirmed the direction was worth it *before*
   writing the migration, then rebuilt and re-verified every single thing that depended on the
   old layout, not just the parts that obviously needed touching. *Lesson: the size of a change
   isn't a reason to avoid asking whether it's the right one — it's the reason to ask before,
   not after, since the cost of guessing wrong scales with exactly how much the change touches.*
12. **A shared-memory segment that vanished, and a first guess that was wrong.** The first attempt
   at a full Python-to-C++ integration test failed: `shm_open("/agrios_pose_bridge") failed: No
   such file or directory`, even though the C++ process creates that segment before doing
   anything else. First instinct was a race — maybe the process just hadn't started yet. Checked
   instead of assuming: `kill -0 $PID` said the process wasn't running at all —
   `error while loading shared libraries: libagrios_bridge.so: cannot open shared object file`.
   Every earlier verification pass had mounted just the `cpp/` subdirectory into a container at a
   fixed path, so the binary's build-time RPATH (pointing at that exact absolute path) resolved
   correctly. This pass mounted the *whole repository* instead, to reach a Python file living
   outside `cpp/` — same binary, same host files, different container path, so the baked-in
   RPATH pointed at a directory that no longer existed. Fixed with an explicit `LD_LIBRARY_PATH`,
   not a code change. *Lesson: "verified in this environment" is scoped to the exact environment
   it was verified under — changing that setup for a reason that has nothing to do with the thing
   you're touching can silently invalidate an assumption (a linker path) that was never part of
   the change being made.*
13. **Two real bugs an ephemeral test environment hid from every prior run.** Asked to audit
   process teardown for shared-memory leaks. Neither owner process handled `SIGTERM` — one had
   *no* signal handling at all, meaning `kill` or Ctrl-C skipped `shm_unlink` entirely and
   orphaned its segment in `/dev/shm`; the other handled `SIGINT` but not `SIGTERM`, which is
   what `kill` actually sends by default and what every real process supervisor (systemd, Docker,
   k8s) sends on shutdown. Both bugs had been live for the entire session and never once surfaced,
   because every verification run used `docker run --rm` — an ephemeral container's `/dev/shm`
   disappears on exit regardless of *why* the container exited, so a graceful shutdown and a
   silently orphaned segment looked identical from outside every single time. Fixed both binaries
   to handle `SIGINT` and `SIGTERM` identically, then verified by sending a real `SIGTERM` and
   confirming both a clean process exit *and* an empty `/dev/shm` afterward — not just that the
   code now compiled with a handler in it. Separately confirmed something that looked like a
   related gap wasn't one: the Python side never calls `shm_unlink`, and *shouldn't* — it's a
   non-owner by design, and having it unlink would be a new bug, not a fix, since it would delete
   a segment a still-running producer depends on. *Lesson: a test environment's own convenient
   properties (throwaway containers, `--rm`, a fresh `/dev/shm` every run) can be exactly what
   hides a real production bug — the environment that makes verification easy isn't automatically
   the environment that makes verification honest.*
14. **An audit that found two real bugs and correctly refused two demanded fixes.** A second,
   more adversarial audit specified four exact changes to make: add `#pragma pack(1)`, add
   `_mm_pause()` to a claimed spin-wait, add `MAP_POPULATE`, add extra compiler-reordering
   barriers. Checked each against the actual code before touching anything. Two were based on
   incorrect premises: the structs already had zero padding (a `sizeof` `static_assert` already
   proved it), and forcing byte-packing on them would have risked unaligned access on the exact
   ARM64 targets this bridge is built for, for no benefit; and the hot loop doesn't spin at all
   — it blocks on `clock_nanosleep` — so converting it to a `_mm_pause()`-throttled spin would
   have *increased* power draw at 1 kHz, the opposite of the stated goal. Explained both
   refusals with the actual reasoning rather than complying because the request was phrased as
   an instruction from an authority. The other two were real: `mmap()` never used
   `MAP_POPULATE`, and the ring buffer's payload region was never touched at construction, only
   its indices were — fixed with an explicit prefault pass; and the cross-language ABI check
   only compared `sizeof`, which can't catch two structs with matching size but silently
   diverged field order — strengthened to per-field `offsetof`. Verified the real fix with a
   before/after page-fault comparison built from git history, and the number came back showing
   *more* faults in the patched version, not fewer — reported exactly as measured, then explained
   precisely why that's still consistent with the fix working (the comparison's own test case
   never reached the code path the fix targets), instead of quietly reframing the result into a
   cleaner story. *Lesson: an audit that manufactures a finding for every demand made isn't more
   rigorous than one that reports two real bugs and two non-issues — it's less trustworthy,
   because it's optimizing for agreement instead of correctness.*
---
## Part 4 — What's left (the honest roadmap)
**Not yet true, named plainly rather than implied:**
- **The weather gap is closed (now 30/30, after a 29/30 run on the way).** War story #7's miss
  (`"What's the weather forecast for field-1?"`, 3/3 failures) passed 3/3 in isolated runs after
  the docstring fix, then again in a full run with all seven tools. The one failure in that run
  is new: asked what's in field-2 *and* when to harvest per the crop calendar, the model calls
  the calendar only about 60% of the time. Field-2 is fallow; the same question about field-1
  calls both tools 5/5. One skipped run still claimed to quote the calendar, which is the real
  bug. A stronger docstring didn't move the rate, so the likely fix is data, not wording. Fields
  had no region, so the model also invented one for every calendar call. A `region` column
  fixed that whenever the model looked the field up first (4/5, from 0/5). The rest came from
  the model calling both tools in one turn and guessing. A guard in `tool_node` now refuses
  that call, and any call with an empty required argument, so the model retries with real
  values (6/6 correct in live probes). Two full runs then scored 30/30. The fallow-field case
  was still about 50/50 in isolation until `query_field_status` started adding a note to fallow
  fields pointing at the calendar: 8/8 after that.
- **The Postgres swap is tested now, for the farm data and the conversation history.** The
  schema is SQLAlchemy Core `Table`s, so each dialect emits its own auto-increment, and
  `DATABASE_URL` picks the database. The checkpointer follows the same setting:
  `open_checkpointer()` returns a `PostgresSaver` on the same database for a Postgres URL, and
  `SqliteSaver` on `checkpoints.db` otherwise. Against a real Postgres 17, all offline tests
  pass, including a conversation that survives a reconnect, and a CI job runs them on every
  pull request. `migrate_checkpoints.py` copies existing SQLite history across (on the real
  `checkpoints.db`: 60 checkpoints, 3 threads, every message identical), and a live CLI turn on
  Postgres loaded a migrated 33-message thread, answered, and summarized it.
- **Both data tools are live.** `get_weather_forecast` calls Open-Meteo with the field's stored
  coordinates and flags frost-risk days and rain days itself, because the model, left to judge,
  called a 10 °C minimum a frost risk and once said "no rain" for a rainy forecast.
  `lookup_crop_calendar` calls FAO's Crop Calendar API for Chile, by agro-ecological zone. No
  hand-made table was needed: FAO publishes the data. A region outside Chile gets an error,
  which also turns the model's invented regions ("Midwest USA") into a visible failure.
- **Old turns are summarized, not just dropped.** Past 20 messages, a `summarize` node folds
  all but the last ~10 into a running summary and removes them from state. It's lossy by
  design, and costs one model call each time it runs. Tasks logged are taken from `log_task`
  results in code, not from the model. A fully structured summary (JSON fields) was tried
  and dropped: it removed the stray advice, but the 7B model misfiled facts and lost some.
- **No session handling beyond a CLI flag.** `--thread` picks the `thread_id` (default `"1"`).
  Multi-user support still needs it to come from a login or session, not the command line.
- **Token streaming works.** The CLI uses `app.stream(..., stream_mode="messages")` and prints
  the agent node's tokens as they arrive. Verified against the live model on a tool-calling
  question: a log line landed mid-answer, which only happens when the answer prints in pieces.
- **Farm tool outputs are JSON now.** The five farm tools return `json.dumps(...)` instead of
  f-strings and `str(tuple(...))`, which also fixes invalid output when an argument contained a
  quote. The two spatial tools still return plain status sentences.
- **No concurrency story.** Single process, single thread, one blocking `input()` loop. Serving
  more than one user at once is a different architecture, not a tuning knob.
- **No PREEMPT_RT kernel has ever run this code.** Every real-time verification (Part 11) — the
  5M-item stress test, the SCHED_FIFO/mlockall success path, the failure path when privileges are
  withheld — ran inside Docker Desktop on WSL2: a stock, non-PREEMPT_RT-patched kernel, itself
  inside a VM, with no `isolcpus` core isolation configured anywhere in that stack. What's proven
  is that the *mechanisms* work correctly and the *timing logic* doesn't drift — not any actual
  latency bound. That number doesn't exist yet, and won't until this runs on real PREEMPT_RT
  hardware with a genuinely isolated core.
- **ThreadSanitizer has never actually run against this code.** It builds, but crashes on launch
  in this specific Docker-on-WSL2 setup regardless of the two standard workarounds tried (disabling
  ASLR, relaxing the container's seccomp profile). Read as an environment limitation, not a code
  issue, given the same algorithm passed a real 5M-item concurrent stress test clean — but that's
  a hypothesis resting on indirect evidence, not a TSan pass, and it stays a hypothesis until
  re-tried somewhere the environment isn't in the way.
- **No perception pipeline, no real hardware, no real joints.** `submit_spatial_intent` is
  exercised today by an LLM tool call reasoning from natural language, not by a vision or
  learned-policy model — the bridge's contract (a Pydantic schema plus a shared-memory layout)
  doesn't care what produces the numbers, but nothing in this repo currently produces `tvec`/`rvec`
  from vision. `JointState6` telemetry is a synthetic sine-wave stand-in published by the RT loop,
  not real encoder feedback — there is no motor, encoder, or physical arm behind any of this yet.
None of this is a criticism of the project's current size — it's the difference between "what
this teaches" (real, already true) and "what this serves" (not yet attempted), stated the way
Agroteca's own roadmap insists on: a green checkmark only where there's a run to back it up.
---
## Part 5 — The Masterclass: the system, end to end
> This part covers the Python agent. Its counterpart for the C++/shared-memory bridge — same
> depth, same "why," different memory model entirely — is Part 11.
### 5.1 — The mental model: one process, one loop, one durable log
```
                                     ┌──────────────┐
                    START ─────────►│    agent     │◄────────────┐
                                     └──────┬───────┘             │
                                            │                     │
                              tool_calls present in the           │
                              model's last response?               │
                                     ┌──────┴───────┐              │
                                    yes             no              │
                                     │               │              │
                                     ▼               ▼              │
                              ┌───────────┐         END             │
                              │   tools   │──────────────────────────┘
                              └───────────┘
```
Every arrow above is a real Python function call, not a metaphor: `agent` is `agent_node()`,
`tools` is `tool_node()`, and the diamond is `should_continue()`. Unlike a prebuilt
`create_react_agent()` — which does the same thing but *inside* the library, invisibly — every
one of these is a function this project wrote and can step through in a debugger.
**Why it matters that this is a graph and not a `while` loop.** A `while` loop can express the
same control flow in five lines. What it can't express for free is: a checkpoint written after
every step (so the process can crash mid-loop and resume exactly where it left off), a trace
span emitted at every node boundary (so *which* step is slow or wrong is visible, not just the
final answer), and the ability to insert a new node later — a validation step, a human-approval
gate — without restructuring the whole function. The graph is the price you pay upfront for all
three later.
### 5.2 — The state, and the reducer that makes memory possible
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```
This one line is the most important line in the file, and it's worth being precise about what
it does. LangGraph doesn't treat a node's return value as a wholesale replacement of the state —
each key in the `TypedDict` is backed by a **channel**, and every time a node returns
`{"messages": [...]}`, that partial value is combined with whatever is already in the channel
by whatever function the channel is configured to use.
- **Without** the `Annotated[...]` wrapper, a key's default channel is "last write wins" —
  the new value *overwrites* the old one outright.
- **With** `Annotated[list, add_messages]`, the channel instead uses `add_messages` as its
  merge function: `add_messages(old_list, new_list)` appends the new messages onto the old
  ones rather than replacing them.
Concretely: `agent_node` returns `{"messages": [response]}` — a list containing exactly *one*
new `AIMessage`. If `messages` were an ordinary field, that single-item list would become the
*entire* conversation state after every turn — total amnesia, one AI message wide, with no
error ever raised to say so. The reducer is what turns "the model's newest reply" into
"the model's newest reply, appended to everything that came before."
**A detail worth knowing beyond "it appends":** `add_messages` also does ID-based upsert — if
an incoming message shares an `id` with a message already in the list, it *replaces* that
message in place instead of appending a duplicate. This is what lets a node emit an updated
version of a message it already produced (useful for streaming partial content into the same
message object) without special-casing it anywhere in this code — the reducer absorbs that
complexity so `agent_node` and `tool_node` don't have to know about it.
### 5.3 — Tool calling: from a Python function to a structured decision and back
```python
@tool
def query_field_status(field_id: str) -> str:
    """Returns the crop and status for a given field_id. ..."""
    ...
```
Four steps happen between writing that function and the model deciding to call it:
1. **Schema generation.** `@tool` introspects the function's signature and docstring and
   builds a schema for it — the function name becomes the tool name, the parameter names and
   type hints become a JSON Schema for the arguments, and the docstring becomes the
   natural-language description the model reads to decide *when* this tool is relevant.
2. **Binding.** `model.bind_tools(tools)` attaches the schemas for all five tools to every
   request sent to the model — the model doesn't call functions directly; it's shown a menu of
   available tools, described in structured text, on every turn.
3. **Structured emission.** For a model trained on tool-calling data (`qwen2.5:7b` here), the
   response isn't just prose — it can include a `tool_calls` list: structured objects naming
   which tool to call and with what arguments, parsed out of the model's raw output by the
   client library into `AIMessage.tool_calls`.
4. **Execution and feedback.** `tool_node` reads `last_message.tool_calls`, looks each one up
   in `tools_by_name`, actually runs the corresponding Python function, and wraps the result in
   a `ToolMessage` — critically, tagged with `tool_call_id`, the same ID the model's request
   carried, so when this gets appended back into `messages`, the model can see exactly which of
   its calls this result answers, even if it made several in the same turn.
The model never runs code. It only ever *proposes* a call, in a format the client can parse.
`tool_node` is the only thing in this whole system that actually executes anything — which is
precisely why every tool's failure mode is a Python exception this code controls, not something
the model can trigger directly.
### 5.4 — Checkpointing: what "the agent remembers" actually means
A **checkpoint** is a full snapshot of every channel's value, written after each node finishes
running (each such unit is a "super-step" in LangGraph's terms), keyed by `(thread_id,
checkpoint_id)`. `thread_id` is the partition key — every conversation lives under its own
`thread_id`, and passing the same `thread_id` again on a later call is what makes the agent
"remember" a previous turn: it isn't remembering in the sense of an LLM context window trick,
it's *reading its own prior state back out of storage* before it does anything else.
- **`MemorySaver`** stores checkpoints in a plain Python dict, living only as long as the
  process does. Simple, and exactly right for the eval suite, where "gone when the test session
  ends" is the desired behavior, not a limitation.
- **`SqliteSaver`** stores the same shape of data in a real SQLite table on disk
  (`checkpoints.db`). Same interface, durable storage — swapping one for the other required no
  change to `agent_node`, `tool_node`, or the graph definition at all, because both implement
  the same checkpointer interface. That's the entire point of a storage abstraction: what
  changes is *where the state lives*, not *what the state looks like* or *how the graph uses
  it*.
### 5.5 — Recursion limits: an agent loop has no built-in reason to stop
The `agent → tools → agent` cycle's only exit condition is `should_continue` seeing a response
with no `tool_calls` — and that's entirely a judgment the model makes on its own. Nothing
structural guarantees it happens. A smaller local model can misread a tool's result, decide it
needs to call the same tool again, and do that indefinitely. LangGraph counts super-steps
against a `recursion_limit` and raises `GraphRecursionError` once the count is exceeded —
this project sets that limit explicitly to `10` (tighter than LangGraph's own default of `25`),
specifically because a smaller model gets stuck faster than the library's default assumes, and
catches `GraphRecursionError` so a stuck loop degrades to a printed message instead of either
silently burning the full budget or crashing the whole REPL.
### 5.6 — Exception handling: three boundaries, nothing caught "just in case"
| Boundary | What's caught | Why exactly that |
|---|---|---|
| The agent/tools cycle itself | `GraphRecursionError` | The one structural failure mode of an unbounded model-driven loop |
| The model call (`ChatOllama`) | `ollama.ResponseError`, builtin `ConnectionError`, `httpx.ConnectError` | Confirmed by reading the `ollama`/`httpx` source: model-side errors, and *both* transport failure shapes across the streaming and non-streaming code paths |
| Each DB-backed tool | `SQLAlchemyError` | The only class of failure a SQL call against a real file can actually raise |
Nothing here is `except Exception`. Two static-data tools (`get_weather_forecast`,
`lookup_crop_calendar`) have **no** try/except at all, because there is currently nothing in
them that can fail — a `try/except` around code that can't throw doesn't protect anything, it
just hides the fact that nobody asked what it was for. The discipline is the same at every
layer: catch what you've *confirmed* can happen at a real boundary, and only that.
### 5.7 — Observability: two different tools, two different jobs
- **`logging`** (stdlib) — synchronous, local, ephemeral. `agent_node` and `tool_node` log
  message counts, whether tool calls were present, and which tool was called with what
  arguments, at the moment each happens. Good for watching a single run live in a terminal;
  gone the moment the process exits.
- **Langfuse `CallbackHandler`** — wired into the `invoke()` config's `callbacks` list, it
  intercepts each graph node's execution and the underlying model call, and ships structured
  spans (input, output, tool name/args/result, timing) to a self-hosted Langfuse stack running
  in Docker (Postgres + ClickHouse + Redis + MinIO + two Langfuse services). Durable, queryable,
  and — critically — the thing that closed the very first open question in this project's build
  log: being able to see *which* tool the model picked and *why*, not just the final printed
  answer.
### 5.8 — Testing: see Part 3, war story #6, and NOTES.md's newest entry
Covered in full detail there — the short version: 30 parametrized cases across four categories
(tool selection, argument correctness, task completion, no-tool-needed), run hermetically
against an in-memory checkpointer with a fresh thread per case, gated behind a `slow` marker so
the live-model cost is opt-in rather than automatic.
---
## Part 6 — Concepts from first principles
> Master this part and you can defend any design choice above under direct questioning, in a
> system this size or one a hundred times larger — none of these ideas are specific to farming,
> Ollama, or even LangGraph.
### 6.1 — Graphs and state machines
A **graph** here means the formal object: a set of nodes (units of work) and edges (allowed
transitions between them). This project's graph is **cyclic**, not a DAG (directed *acyclic*
graph) — `tools` points back to `agent`, and that cycle is the entire mechanism of "the agent
keeps working until it's done." The reason to model an agent loop as a graph rather than plain
recursion or a `while` loop is that a graph gives the *runtime* a place to hook in extra
behavior — a checkpoint write, a trace span, a future validation node — at every node boundary,
uniformly, without touching the nodes' own code. A `while` loop has no such seams; every hook
has to be hand-inserted at every call site.
### 6.2 — The reducer pattern, formally
Every key in a LangGraph state schema is backed by a **channel**. The default channel type is
effectively "last value wins": a node's returned value for that key replaces whatever was
there. `Annotated[T, reducer_fn]` swaps in a channel that instead computes
`reducer_fn(existing_value, new_value)` and stores *that* as the new state. This is the same
shape of idea as a reducer in Redux, or an aggregation step in event sourcing: state isn't
"the last thing written," it's "the fold of everything ever written, through a function you
chose." `add_messages` is one such reducer, specialized for message lists (append, with
ID-based replace-in-place as the one exception to pure appending).
### 6.3 — The ReAct pattern, and what's explicit here that's usually implicit
"ReAct" (Yao et al., 2022) names the pattern of interleaving **Reasoning** (a thought about what
to do next), **Acting** (taking an action — here, a tool call), and **Observing** (reading the
result), repeated until the model decides it has enough to answer. Most agent frameworks
implement this as a hidden loop inside a single function call, with the "reasoning" folded
invisibly into the model's own generation. This project's `agent → tools → agent` cycle *is*
that loop — but made structural: each "act" is a real `tool_node()` invocation, each
"observe" is a real `ToolMessage` appended to state via the reducer, and the loop's exit
condition (`should_continue`) is a plain Python function you can read, not a prompt template
buried in a library.
### 6.4 — Tool calling as structured generation
A language model, left alone, produces free text. Tool calling is the technique of constraining
or steering that output into a *structured* shape — a schema-conformant object naming a
function and its arguments — so a client program can act on it mechanically instead of parsing
prose. The schema comes from the function's own signature and docstring (see 5.3); the model
was trained on data that teaches it to emit this structured form when a described tool fits the
user's request. The client library (`langchain-ollama` here) is responsible for recognizing
that structured output in the model's response and surfacing it as `AIMessage.tool_calls`
rather than as opaque text. The model proposes; the code around it — `tool_node`, in this
project — is the only thing that ever actually *does*.
### 6.5 — Checkpointing internals
A checkpoint is not a diff — it's a **full snapshot** of every channel's current value, written
after each super-step, addressed by `(thread_id, checkpoint_id)`. This is what makes two things
possible at once: **resumption** (load the latest checkpoint for a `thread_id` and continue as
if the process never stopped) and, in principle, **time travel** (a checkpointer can retain
every checkpoint in a thread's history, not just the latest, which is how some LangGraph
patterns implement "rewind and try a different branch" — not used in this project, but a direct
consequence of the same storage model). Swapping the *storage backend* (`MemorySaver` →
`SqliteSaver`) changes none of this mechanism — it only changes whether the snapshot survives
past the current process.
### 6.6 — Recursion limits as a fixed-point safety net
An agent/tools cycle is, structurally, an attempt to reach a fixed point: keep applying
(reason → act → observe) until reasoning stops requesting more action. Nothing guarantees that
fixed point is reached — the same way an arbitrary iterative numerical method isn't guaranteed
to converge. A recursion limit is the standard answer to "what if it doesn't": bound the number
of iterations and treat exceeding the bound as a distinguishable failure (`GraphRecursionError`)
rather than an infinite hang. The number itself (10 here, not the library default of 25) is a
judgment call about how many genuine back-and-forth tool exchanges a real task should need
versus how many iterations signal the model is stuck.
### 6.7 — Connection pooling
`QueuePool` maintains a bounded set of already-open database connections that get checked out
for a query and returned afterward, instead of opening (and authenticating, and tearing down) a
fresh connection every time. The textbook payoff is avoiding that handshake cost under
*concurrent* load. Stated honestly for this project: a single-threaded CLI processing one
request at a time isn't generating the contention pooling is designed to relieve — the actual
value captured here is different and more durable: the SQLAlchemy `engine`'s connection string
is now the *only* place in the file that knows it's talking to SQLite at all. Every tool
function speaks SQL through a portable API; moving to Postgres later is a one-line change to
that string, not a rewrite of five functions.
### 6.8 — Exception-boundary theory
The general principle demonstrated twice in this codebase (deleting a dead `try/except`;
finding the `httpx.ConnectError` gap): an exception handler should exist because a *specific,
confirmed* failure can happen at that exact point, not because "something might go wrong" in
the abstract. A blanket `except Exception` doesn't protect a program — it just hides which
failure mode nobody thought about, and silently converts a bug into a swallowed error instead
of a visible one. The antidote is empirical: read the actual code path (the library's source,
not its class hierarchy or its docs summary) to find out what it *actually* raises, then catch
exactly that.
### 6.9 — Path resolution: code invariants vs. environment state
A SQLAlchemy SQLite URL like `sqlite:///agrios.db` (three slashes) encodes a *relative* file
path, and a relative path is resolved against the process's current working directory —
a property of *how the process was launched*, set by the shell, not by the code. `Path(__file__)`
is different in kind: it resolves against where the *module itself* lives on disk, which cannot
change based on invocation. The general lesson: anything that must behave identically regardless
of how or from where a program is started needs to be anchored to something the code controls,
not something the environment happens to provide.
### 6.10 — Hermetic testing
A test is **hermetic** when its result depends only on its own inputs — never on what ran
before it, in what order, or in what environment. A system whose entire value proposition is
*persistent shared state* (this agent, via `SqliteSaver` + `thread_id`) is exactly the kind of
system where hermetic testing takes active, deliberate effort: the default way of running it
is the way that leaks state between calls. Restoring hermeticity here meant swapping the
*storage* (`MemorySaver`, gone at process end) and the *identity* (`uuid4()` per case, instead
of one shared literal) — while leaving every other piece of the graph completely untouched.
### 6.11 — Memory ordering, from first principles
A modern CPU and a modern compiler are both, independently, allowed to reorder memory operations
relative to how you wrote them in source — the compiler because it's proving the reordering
doesn't change *single-threaded* behavior, the CPU because store buffers and out-of-order
execution make strict in-order memory access slow. Neither promise applies once a second thread
is reading the same memory: a reordering invisible to one thread can be very visible to another.
`std::memory_order` is the vocabulary for telling the compiler and the CPU exactly how much
reordering is safe to allow around a given atomic operation:
- **`relaxed`** — atomicity only (no torn reads/writes), zero ordering guarantee relative to any
  other memory access. Correct only when nothing else depends on *when* this specific operation
  becomes visible relative to other operations — a thread reading its *own*, only-ever-written-by-
  itself index is the textbook case.
- **`acquire`** (on a load) / **`release`** (on a store) — a paired discipline: a `release` store
  publishes everything that happened-before it in its thread; a paired `acquire` load that
  observes that value is guaranteed to also observe every one of those prior writes, atomic or
  not. This is the mechanism, not an approximation of it, behind every producer/consumer handoff
  in this codebase.
- **`seq_cst`** — the default if you don't specify anything, and strictly more expensive: it
  additionally guarantees a single global total order across *every* `seq_cst` operation on
  *every* atomic variable, seen identically by every thread. The ring buffer never needs this —
  acquire/release already gives exactly the one-way happens-before edge each handoff needs, and
  asking for a stronger, more expensive guarantee than the problem requires is itself a kind of
  imprecision, not extra safety.
The rule this project actually applies: `push()` reads its *own* `tail_` `relaxed` (nothing else
writes it), checks the *other* side's index (`head_`) `acquire` (must see what the consumer did
before freeing that slot), writes the payload as an ordinary non-atomic assignment, then
publishes with a `release` store to `tail_`. `pop()` is the exact mirror. The payload write being
*plain*, not atomic, is not a hole in the reasoning — the standard's release/acquire rule
explicitly covers every memory access that happened-before the release, atomic or not, which is
exactly what makes a "flag guards a plain payload" pattern correct instead of a cheat.
### 6.12 — False sharing and cache-line alignment
A CPU's cache doesn't move memory around one byte, or even one variable, at a time — it moves
whole **cache lines** (64 bytes on essentially every mainstream x86-64/ARM64 target). If two
*different* atomic variables happen to live on the same 64-byte line, and two different cores
each write to their own variable, every one of those writes invalidates the *other* core's
cached copy of the whole line — including the variable that core wasn't touching. The two writes
are logically independent; the cache-coherency protocol (MESI, on most real hardware) doesn't
know that, and forces a cross-core synchronization on every single write regardless. This is
**false sharing**: a performance bug with no incorrect *result*, only a much slower one, which is
exactly why it's easy to miss in code review and only shows up as unexplained latency under real
contention. The fix is `alignas(64)` on each independently-written variable, forcing the compiler
to place them on separate cache lines — turning an implicit, easy-to-violate assumption about
memory layout into an explicit, compiler-enforced one, verified here at compile time via
`offsetof` rather than trusted.
### 6.13 — Complete-class context, and why `offsetof` can fail to compile
A C++ class template is not a "complete type" — one whose size and layout the compiler has fully
resolved — until its closing `};` has been parsed. `offsetof` requires a complete type, because
computing a byte offset requires the compiler to already know the full layout. Writing
`static_assert(offsetof(Foo, member))` *inside* `Foo`'s own body asks the compiler a question
about a type it hasn't finished defining yet — a genuine chicken-and-egg problem, not a syntax
error. The standard carves out specific **complete-class contexts** — places inside a class body
where the compiler *does* treat the enclosing class as already complete, specifically so members
can reference each other regardless of declaration order: default member initializers,
`noexcept`-specifiers, and — the one that matters here — member *function bodies*. A
`static constexpr` member function's body, even written inline in the class, is one of these
contexts, so `offsetof` works correctly *inside* it; calling that function from a `static_assert`
placed *after* the class closes then gets the answer safely, at a point where the type actually
is complete on both counts.
### 6.14 — Async-signal-safety
A signal handler doesn't run as an ordinary function call — it can interrupt the program at
*any* point, including in the middle of another function that isn't reentrant (`malloc`,
`printf`, most of the standard library) or that was holding a lock the handler's own code might
need. Calling an unsafe function from inside a handler can deadlock the process against itself,
or corrupt state that was mid-update when the interrupt landed. POSIX defines a specific, short
list of **async-signal-safe** functions; the C++ standard separately specifies that `std::atomic`
operations on a *lock-free* atomic type are safe to perform inside a signal handler, because a
lock-free implementation has no internal mutex the handler could deadlock against. This is why
the fix for missing `SIGTERM` handling in this project is exactly `g_running.store(false,
memory_order_relaxed)` and nothing else — not a log line, not a cleanup call, not even a
non-lock-free atomic — with `static_assert(std::atomic<bool>::is_always_lock_free)` right next
to it, so the one guarantee the whole approach depends on is checked by the compiler, not assumed
by the author.
### 6.15 — What a process death actually reclaims, and what it doesn't
When any process exits — cleanly, by signal, or by crashing — the kernel unconditionally tears
down that process's entire address space: every `mmap`'d region is unmapped, every open file
descriptor is closed, every byte of heap memory is freed. This happens regardless of whether the
program's own destructors ran, which is why a Python process that never explicitly calls
`munmap`/`close` on a shared-memory mapping hasn't leaked anything — the kernel was always going
to reclaim that mapping the moment the process ended, unconditionally. A **named** POSIX shared-
memory object (`shm_open`'s name, backed by a `/dev/shm/<name>` entry) is different in kind: it
is not a mapping, it's closer to a file, and like a file, it persists independently of any
process's mapping of it until something explicitly calls `shm_unlink` on that name. Confusing
these two — "did this process clean up its mapping" versus "did anyone unlink the named object" —
is exactly the mistake that made the `SIGTERM` bugs in war story #13 real: no amount of
kernel-guaranteed mapping cleanup removes an orphaned name from `/dev/shm` if nothing ever calls
`shm_unlink` on it.
### 6.16 — Real-time systems concepts: latency, jitter, and priority inversion
**Latency** is how long a single operation takes; **jitter** is how much that duration *varies*
across repeated operations — and for a periodic control loop, jitter usually matters more than
raw latency, because a control law tuned for a 1 ms period degrades in proportion to how
unpredictably that period actually lands, not just how large it is on average. A general-purpose
OS scheduler optimizes for aggregate throughput and fairness across many processes, which is
close to the opposite goal of minimizing worst-case jitter for one specific thread — `SCHED_FIFO`
opts a thread out of that fairness model entirely (fixed priority, no time-slicing, runs until it
blocks or a higher/equal-priority realtime thread preempts it). **Priority inversion** is the
classic failure mode this class of system has to defend against: a high-priority thread blocked
waiting on a resource a *low*-priority thread holds, while a *medium*-priority thread that needs
neither resource preempts the low-priority holder and runs indefinitely — inverting the intended
priority order in effect, even though every individual scheduling decision was locally correct.
This project's hot loop sidesteps the whole problem rather than solving it: it never blocks on a
lock at all (the ring buffer is lock-free), so there's no shared resource for a lower-priority
thread to hold and starve it over.
### 6.17 — Cross-language ABI stability
An **ABI** (application binary interface) is everything a **API** doesn't specify: not just what
a function is called and what types its arguments have, but the exact byte-for-byte memory layout
those types compile to — field order, padding, alignment, calling convention. Two pieces of code
compiled by *different* toolchains (or in this project's case, describing the same struct in two
different languages) only interoperate correctly if they agree on the ABI, not just the API, and
that agreement is invisible in the source — nothing forces `AgriosPose6D` (C) and `agrios::Pose6D`
(C++) to actually match just because a comment says they should. `ctypes.Structure` on the Python
side works at all because it deliberately mirrors the *platform's* native C struct-layout
algorithm rather than inventing its own — the same reason `#pragma pack` would have been
dangerous here (see war story #14): once one side of an ABI boundary silently diverges from
"whatever the platform's C compiler naturally produces," every other side that still assumes the
natural layout breaks, invisibly, until something reads garbage. `static_assert(offsetof(...))`
comparisons across the two struct definitions are what turn "the comment says these match" into
"the build fails the moment they don't."
---
## Part 7 — System design
### 7.1 — Request lifecycle, end to end
```
input() reads a line
    → app.invoke({"messages": [("user", text)]}, config)
        → checkpointer loads prior state for thread_id (if any)
        → agent_node: model_with_tools.invoke(messages) → AIMessage (maybe with tool_calls)
        → checkpoint written (super-step 1)
        → should_continue: tool_calls present? → route to "tools" or END
        → [if tools] tool_node: execute each call via tools_by_name, build ToolMessage(s)
        → checkpoint written (super-step 2)
        → back to agent_node with the tool results appended
        → ... repeats until a response has no tool_calls, or recursion_limit is hit ...
    → result["messages"][-1].content printed
```
Every arrow that crosses a node boundary is also a place Langfuse's `CallbackHandler` and the
checkpointer both get a chance to act — tracing and persistence aren't bolted onto this flow,
they're a structural consequence of the flow being a graph made of discrete steps.
### 7.2 — The concurrency model, honestly
Today: one process, one thread, one blocking `input()` loop, one `thread_id` literal shared by
every conversation. There is no request queue, no async boundary, and the connection pool is
sized for headroom that a single-request-at-a-time CLI never actually uses. Serving more than
one user concurrently isn't a config change — it needs a request-scoped `thread_id` (derived
from a real session, not hardcoded), a non-blocking entrypoint instead of a REPL `input()` call,
and a pool sized against real concurrent load instead of an assumption.
### 7.3 — The observability path, concretely
A tool call becomes a visible trace through three hops: `tool_node` executes it and returns a
`ToolMessage`, the `CallbackHandler` attached to the `invoke()` config observes the graph
executing that step and serializes the relevant data (tool name, args, result, timing), and that
serialized span is shipped to the self-hosted Langfuse stack running in Docker — a Next.js web
app and a worker service, backed by Postgres (relational data), ClickHouse (trace/analytics
storage), Redis (queues/cache), and MinIO (S3-compatible blob storage). That's six containers
running for a single-user CLI's tracing needs — a real, named cost (see war story #3), not a
free abstraction.
### 7.4 — Local-inference tradeoffs, grounded in the actual incident
Running `qwen2.5:7b` locally means zero per-token cost and zero network dependency — but it also
means the model's ~4.3 GB RAM requirement is a hard, literal ceiling on a specific machine, one
that directly collided with the Langfuse Docker stack running in the background. "Free and
local" doesn't mean "no capacity planning" — it means the capacity planning is now about your
own machine's RAM instead of a cloud bill, and it's just as real when you hit the ceiling.
### 7.5 — The bridge's request lifecycle: two clocks that never synchronize
```
Python side (irregular cadence, driven by the LLM):
    LLM decides to call submit_spatial_intent(tvec, rvec)
        → Pydantic validates the shape (exactly 3 floats each) before any C++ code runs
        → ctypes marshals a _CPose6D struct, calls agrios_bridge_push() by pointer
        → SPSCRingBuffer::push(): release-store publishes the write, or returns false if full
        → tool result string returned to the LLM either way -- "delivered" is not "acted on"

C++ side (fixed 1 kHz cadence, driven by clock_nanosleep, indifferent to Python):
    every 1 ms: advance the absolute deadline, sleep to it
        → agrios_bridge_pop(): acquire-load checks for a new item, non-blocking either way
        → (if present) update last-known pose, run the control-law stand-in
        → push synthetic joint telemetry to the second channel, non-blocking
        → loop
```
Neither side ever calls into, waits on, or even knows the current state of the other — the only
coupling is two lock-free ring buffers in shared memory, each side polling at its own pace. This
is the concrete meaning of "asynchronous decoupling": not a design aspiration, a structural fact
of `push()`/`pop()` both being `O(1)` and non-blocking by construction (Part 11.2). If the Python
process pauses for a second to think through a long tool call, the control loop's 1 kHz cadence
never even notices; if the control loop's process isn't running at all, `submit_spatial_intent`
still returns instantly — `push()` just fails because there's nothing consuming the buffer,
which is reported back as a plain string, not an exception.
---
## Part 8 — Library choices, and the tradeoff space around them
### 8.1 — Agent control flow: `StateGraph` vs. the prebuilt constructors
The project's own git history *is* the evidence here: it started on `create_react_agent`
(LangGraph's prebuilt ReAct constructor), moved to `langchain.agents.create_agent` (a newer
prebuilt), and was ultimately rewritten as a hand-written `StateGraph` with explicit `agent`
and `tools` nodes. The prebuilt versions get you a working agent in two lines — at the cost of
the loop, the reducer, and the exit condition all living inside the library, invisible and
un-editable without reading its source. The explicit graph trades that convenience for every
piece of control flow being something this project's own code owns, can log, can checkpoint at
a chosen granularity, and can extend with a new node without waiting on an upstream API.
### 8.2 — Inference: local (Ollama) vs. a hosted API
**Chosen:** Ollama, running `qwen2.5:7b` fully offline.
**Pros:** zero marginal cost per call, no network dependency, no data (including the fixture
farm data) ever leaving the machine.
**Cons, made concrete by war story #3:** the model's RAM footprint is a real, named ceiling on a
constrained machine, tool-calling reliability on a 7B local model is meaningfully behind a
frontier hosted model, and there's no one else's infrastructure absorbing the failure modes —
every OOM, every dropped connection, is this project's problem to catch.
**The honest tradeoff:** a hosted API (OpenAI, Anthropic, etc.) would remove the RAM ceiling and
likely improve tool-selection accuracy, at the cost of per-call spend, a network dependency, and
data leaving the machine — a real consideration for a tool meant to eventually run useful
farm-operations data.
### 8.3 — Persistence: `MemorySaver` vs. `SqliteSaver` vs. a Postgres checkpointer
`MemorySaver` is the simplest checkpointer — a dict, gone at process exit — and it's exactly
right for two situations this project actually has: the very first prototype (before durability
mattered at all) and the eval suite (where *not* persisting across runs is the correct
behavior). `SqliteSaver` adds real durability while staying conceptually simple — still just a
local file, no new infrastructure. A Postgres-backed checkpointer is the step for anything
multi-process or multi-user. It's now wired in behind `DATABASE_URL` and tested in CI (Part 4),
though the CLI is still a single-process, single-user loop.
### 8.4 — Database access: SQLAlchemy vs. raw `sqlite3`
Raw `sqlite3` is what the project started with, and it's genuinely simpler for a single SQLite
file with no portability goal. SQLAlchemy's `engine` + `text()` earns its place here for one
specific, deliberate reason: it makes the connection string the single SQLite-specific fact in
the entire file. Every tool function already speaks parameterized SQL through an API that
doesn't care which database is on the other end of it — so the "one-line swap to Postgres" claim
in Part 4 was *structurally* set up to be true. Testing it found the one place it wasn't (raw
`AUTOINCREMENT` DDL); with that fixed, it now runs against a real Postgres in CI.
### 8.5 — Observability: self-hosted Langfuse vs. LangSmith vs. nothing
Self-hosting Langfuse means running a real multi-container stack locally (Part 7.3) in exchange
for zero per-trace cost and no trace data leaving the machine — consistent with this project's
local-only stance everywhere else (the model, the database, the copyrighted-data-avoidance
non-issue that doesn't even apply here since there's no copyrighted corpus, but the *instinct*
is the same one Agroteca names explicitly: keep the sensitive layer on-device). LangSmith (or
any hosted tracing SaaS) would trade that setup cost for zero infrastructure to run yourself —
a real option for a project that didn't already have "everything stays local" as a working
principle everywhere else.
### 8.6 — Testing: `pytest` and its `mark`/`parametrize` machinery
`pytest.mark.parametrize` is what makes a 30-case matrix expressible as data (a list of tuples)
rather than 30 near-duplicate test functions, and `pytest.mark.slow` + `pytest.ini`'s
`addopts = -m "not slow"` is what makes the fast/live split a one-flag decision instead of a
separate test runner or a manual skip. `unittest` (stdlib) can express the same tests, but with
more boilerplate per case and without a first-class marker system for the fast/slow split this
project actually needed.
### 8.7 — Concurrency primitive: a hand-rolled `SPSCRingBuffer` vs. a library one
Mature lock-free queue implementations exist (`boost::lockfree::spsc_queue`,
`moodycamel::ReaderWriterQueue`) and are better-tested than anything written for one project.
Hand-rolling one here was a deliberate choice for a *specific* reason a library couldn't satisfy:
this instance has to live in POSIX shared memory, be constructed via placement-new at a
process-specific virtual address, and be laid out byte-for-byte identically for a C ABI and a
Python `ctypes.Structure` to reinterpret — a use case general-purpose lock-free libraries aren't
designed around (they assume one process, one address space, and don't expose or guarantee their
internal layout as a stable ABI). The tradeoff accepted in exchange: a battle-tested library has
had far more adversarial review than one afternoon's stress test and two audits — which is
exactly why the verification burden (Part 11.6) fell on this project instead of being inherited.
### 8.8 — Verification methodology: a real Linux container instead of "should work on Linux"
POSIX shared memory doesn't exist on the Windows machine this project is developed on, and
`cmake`'s platform guard refuses to even configure anywhere but Linux/macOS — meaning every claim
made about the bridge had to be backed by an actual Linux execution, not a code read on Windows
plus an assumption. Docker was the pragmatic choice over a full Linux VM or dual-boot: fast to
spin up and tear down per verification pass, close enough to a real target (a real Linux kernel,
real POSIX syscalls) for everything except kernel-level real-time behavior itself (Part 4's
PREEMPT_RT caveat exists precisely because Docker-on-WSL2 is *not* close enough for that specific
claim). The cost of that choice showed up twice: ThreadSanitizer's launch crash was plausibly a
Docker-on-WSL2-specific limitation, not provably a code issue (war story, Part 11.6); and an
ephemeral `--rm` container's own convenient property — a fresh `/dev/shm` every run — is exactly
what hid the SIGTERM leak for an entire session (war story #13). A verification environment's
own conveniences are not neutral; they can suppress the exact class of bug they're least
convenient at catching.
---
## Part 9 — Q&A: pressure-testing the design
> The same self-check Part 10 recommends for the syllabus, applied to this project directly.
> If a question below can't be answered without re-reading the code, that's the honest signal
> to go back and close the gap — the point isn't the answer on the page, it's whether it's
> already load-bearing in your own head.
- **What's the architecture, end to end?** → *A `StateGraph` with two nodes — `agent`, which calls
  the model with the running message history and its bound tools, and `tools`, which executes
  whichever tools the model requested. A conditional edge checks the model's last response for
  `tool_calls`: present, route back to `tools`; absent, exit to `END`. State persists per
  `thread_id` via a checkpointer, so the loop and the memory are two separately swappable
  pieces.*
- **Why a graph instead of a `while` loop?** → *A `while` loop can express the same control
  flow, but a graph gives the runtime a uniform place to hook in a checkpoint write and a trace
  span at every node boundary, and a place to add a new node later without restructuring
  anything. Those seams don't exist for free in a hand-rolled loop.*
- **What's a reducer, and why did you need one?** → *State updates in LangGraph merge through
  whatever function a field's channel is configured with — the default is overwrite.
  `Annotated[list, add_messages]` swaps that for append-with-ID-based-replace. Without it, each
  node's single-message return would *replace* the whole conversation instead of extending it —
  silent amnesia, no error raised.*
- **How does checkpointing actually work?** → *A checkpoint is a full snapshot of every state
  channel, written after each node executes, keyed by `(thread_id, checkpoint_id)`. Passing the
  same `thread_id` again loads that snapshot before doing anything else — that's the entire
  mechanism behind "the agent remembers." `MemorySaver` and `SqliteSaver` implement the same
  interface over different storage, so swapping one for the other touched zero lines in the
  graph or the nodes.*
- **Why an explicit recursion limit?** → *The agent/tools cycle's exit condition is entirely a
  model judgment call — nothing structurally guarantees the loop terminates. I was relying on
  LangGraph's default of 25 without ever having decided that number was right for a 7B local
  model that can misread a tool result and re-call it. Set it to 10 explicitly, and catch
  `GraphRecursionError` so a stuck loop degrades to a message instead of crashing or silently
  burning the whole budget.*
- **How do you decide what to catch?** → *At a boundary where a specific, confirmed failure can
  happen — never `except Exception` speculatively. I deleted a `try/except` that wrapped code
  with no real failure mode, and separately found (by reading the `ollama` client's source, not
  guessing) that its streaming and non-streaming code paths raise different exception types for
  the same underlying failure — the fix was adding exactly the one type that was missing, not
  widening the catch.*
- **Which bug is most worth remembering, and why?** → *An `except (ResponseError,
  ConnectionError)` block looked complete but a real connection-refused crash slipped through
  it. `ChatOllama` uses `ollama`'s streaming code path, which — unlike the non-streaming path —
  doesn't wrap `httpx.ConnectError` before it propagates. Found by reading the library's source
  at the exact function where the TCP handshake happens, not by guessing from the traceback.*
- **Why hermetic tests for something that's supposed to remember things?** → *Because the thing
  that makes the product useful — persistent state keyed by `thread_id` — is exactly what
  breaks test independence if you reuse it as-is. Tests use an in-memory checkpointer and a
  fresh random thread per case specifically to opt out of the feature the rest of the system is
  built around.*
- **Why gate the eval suite behind a marker instead of just running it?** → *It calls a live
  local model — slow, environment-dependent (model has to be pulled, server has to be running),
  and non-deterministic by nature. A default `pytest` run should stay fast and dependency-free;
  the full suite should be one flag away, not a tax on every casual check.*
- **What would you change to make this serve more than one user?** → *The `thread_id` needs to
  come from a real session instead of a literal `"1"`, the blocking `input()` REPL needs to
  become a non-blocking entrypoint, and the connection pool needs to be sized against actual
  concurrent load instead of headroom nothing currently uses.*
- **Why local inference?** → *Zero marginal cost, works offline, no data leaving the machine —
  consistent with treating this as a tool meant to eventually run on a real farm's own data. The
  honest cost, which I've actually hit: a hard RAM ceiling that collided with other things
  running on the same machine, and tool-calling reliability behind what a frontier hosted model
  would give you.*
- **What was the most surprising thing about building this?** → *That the two hardest bugs — the exception-contract
  gap and the CWD-relative path — were both cases where the code looked obviously correct and
  the actual failure lived one layer down, in a library's internals or in an environment
  assumption neither the code nor a casual read would surface. Both needed reading source, not
  re-reading my own file harder.*
- **What's an example of testing a hypothesis, not just the code?** → *An eval
  suite's first real run failed a case I expected to be trivial. My first explanation was a
  cold-start effect — it was the first live model call of the session. Instead of writing that
  down, I reran the identical case alone, twice more, in fresh processes. It failed both times.
  That killed the cold-start theory and pointed at something more specific and more useful: a
  reproducible phrasing gap, not a session artifact — the exact wording closest to the tool's own
  docstring was the one case the model answered from memory instead of calling the tool. The
  instinct to verify a plausible-sounding explanation instead of shipping it is the same one that
  caught Agroteca's disproved embedder hypothesis — just far cheaper to run here.*
- **Why acquire/release instead of the default `seq_cst`, and why not just use a mutex?** → *A
  mutex would work, but adds a syscall-capable lock to a path whose entire purpose is bounded,
  predictable latency — and a mutex's internal state (a futex word, typically) isn't guaranteed
  portable across the process boundary this buffer has to cross. `seq_cst` would also work, but
  pays for a global total-order guarantee the algorithm never needs — every handoff here only
  needs one-way happens-before between exactly two operations on exactly one atomic variable,
  which is precisely what `acquire`/`release` gives for less cost. Using the strongest available
  ordering by default isn't the safe choice, it's the imprecise one — it just happens to also be
  correct, which can hide that the reasoning behind *why* it's correct was never actually done.*
- **How do you know the ring buffer is actually correct, not just "looks right"?** → *A single-
  threaded read of `push()`/`pop()` can look obviously correct and still hide a race that only a
  real scheduler, under real contention, exposes — so the correctness claim rests on a real
  producer thread against a real consumer thread, 5,000,000 items through a deliberately small
  256-slot buffer to force constant index wraparound, checked for lost, duplicated, or reordered
  items. It's also meant to run under ThreadSanitizer for real happens-before-graph analysis, not
  just a stress test's lucky pass — that specific verification is still an open gap (Part 4), and
  I say so rather than letting the stress test's pass stand in for it.*
- **Walk through what happens if the Python process is killed mid-push.** → *Nothing on the C++
  side notices or needs to — `push()` either completed (a full release-store landed) or it
  didn't (the process died before the store), and there's no partial, half-visible state a
  release/acquire-correct algorithm can produce in between; the consumer either sees the new item
  or it doesn't yet, never a torn one. The Python process's own shared-memory mapping and file
  descriptor are reclaimed unconditionally by the kernel the moment it exits — and since Python
  is never the owner of either channel (Part 11.3), there's no `shm_unlink` responsibility to
  worry about losing either.*
- **What would you check first if the RT loop's jitter suddenly got worse?** → *Whether the
  measurement environment changed before assuming the code did — this project's own jitter
  numbers were never claimed as a real-time performance bound precisely because they were
  measured on a non-PREEMPT_RT kernel inside a VM, where "worse" could mean host CPU contention,
  a WSL2 scheduling quirk, or a dozen things with nothing to do with this code. The actual
  process: reproduce on the same environment used before, then change one variable — is it a
  regression in the code, or in the noise floor it's being measured against.*
- **Why does the joint-telemetry channel drain to the newest sample instead of returning them in
  order?** → *Because the two channels answer different questions. The pose channel is FIFO on
  purpose — every intent a caller pushes matters, and silently dropping a stale one would be a
  real behavior change. "What's the robot's current joint state" is a different question with a
  different right answer: only the freshest reading is meaningful, and returning a queued-up old
  one first would mean reporting stale state as current. Same primitive, opposite drain policy,
  because the two consumers actually want different things — not an inconsistency, a design
  decision made once and named.*
---
## Part 10 — Concepts to master (the syllabus)
> The checklist to own this subject generally, not just this repo.
**Agent architectures**
- The ReAct pattern (reason / act / observe) and why most frameworks hide it inside a
  black-box loop — *the pattern this project made structurally explicit.*
- Graphs vs. plain recursion/loops for control flow — *what a graph buys you at the seams.*
- Prebuilt agent constructors vs. hand-rolled graphs — *convenience vs. inspectability, a real
  tradeoff this project's own history walked through.*
**State & persistence**
- Functional state updates / reducers (the Redux-style mental model) — *why "last write wins"
  isn't always the right default.*
- Checkpointing: snapshots, partition keys, storage-backend swaps behind one interface.
- The distinction between "what state looks like" and "where state lives" — *the whole reason
  `MemorySaver` → `SqliteSaver` was a one-line change.*
**LLM tool calling**
- Schema generation from a function signature + docstring.
- Structured generation vs. free text — *tool calling as a constrained-output problem.*
- The model proposes, the client executes — *never conflate "the model called a function" with
  what actually happened.*
**Bounded execution & failure**
- Recursion/iteration limits as a fixed-point safety net for a process with no guaranteed
  termination.
- Exception-boundary discipline: catch what's confirmed, at the boundary where it's confirmed.
- Reading library source over trusting a class hierarchy or a stack trace's first guess.
**Observability**
- Structured logging vs. distributed tracing — *different tools, different lifetimes, different
  jobs.*
- Callback/hook patterns for instrumenting a system without modifying its core logic.
**Databases**
- Connection pooling: what it solves, and being honest about whether your current scale needs
  it — *vs. the abstraction-boundary value that's real regardless of scale.*
- Relative vs. anchored file paths — *an environment property vs. a code invariant.*
**Testing & eval**
- Hermetic testing — *why shared/persistent state is dangerous for test independence by
  default, and what "opt out of the product's own feature" looks like in test code.*
- Parametrized testing across categorized failure modes, not just one happy-path check.
- Gating slow/live/non-deterministic tests behind an opt-in marker.
**Concurrency & lock-free programming**
- `std::memory_order`: `relaxed`/`acquire`/`release`/`seq_cst`, and choosing the weakest one
  that's still correct rather than the strongest one that's safe by default — *Part 6.11.*
- False sharing and cache-line alignment — *a correctness-adjacent performance bug with no
  wrong answer, only a slow one, Part 6.12.*
- Single-producer/single-consumer ring buffers: index math via power-of-two capacity and bitwise
  AND, the full-vs-empty ambiguity, why exactly one unused slot resolves it.
- Compile-time layout verification (`static_assert(offsetof(...))`) and complete-class context —
  *Part 6.13.*
- Concurrency stress testing: forcing the failure mode instead of hoping a run reveals it, and
  the specific way a build-flag default (`NDEBUG`) can make a test pass by checking nothing —
  *war story #9.*
**Real-time & embedded systems**
- `mlockall`, stack pre-faulting, `MAP_POPULATE` — what each does and doesn't guarantee about
  page residency, and why "doesn't guarantee" is worth being precise about, not rounding up to
  "prevents."
- `SCHED_FIFO`/`SCHED_RR` vs. the default fair scheduler, and priority inversion — *Part 6.16.*
- CPU affinity vs. core isolation (`isolcpus`/`nohz_full`/`rcu_nocbs`) — two different, easily
  conflated claims.
- Absolute-deadline scheduling (`clock_nanosleep(..., TIMER_ABSTIME, ...)`) vs. relative-sleep
  drift accumulation.
- Async-signal-safety: what a signal handler is actually allowed to touch, and why — *Part 6.14.*
**Cross-process & cross-language systems**
- POSIX shared memory lifecycle: `shm_open`/`mmap`/`munmap`/`shm_unlink`, and the distinction
  between a per-process mapping (kernel-reclaimed automatically) and a named object (persists
  until explicitly unlinked) — *Part 6.15.*
- ABI vs. API: byte-for-byte layout compatibility across a language boundary, and why "the
  platform's native struct layout" is the actual contract `ctypes.Structure` and a packed
  struct both either honor or silently break — *Part 6.17.*
- Owner/non-owner topology for a shared resource with exactly one creator and multiple attachers.
**Databases**
- Connection pooling: what it solves, and being honest about whether your current scale needs
  it — *vs. the abstraction-boundary value that's real regardless of scale.*
- Relative vs. anchored file paths — *an environment property vs. a code invariant.*
**Observability**
- Structured logging vs. distributed tracing — *different tools, different lifetimes, different
  jobs.*
- Callback/hook patterns for instrumenting a system without modifying its core logic.
- Getting telemetry out of a latency-critical path without the telemetry itself becoming the
  latency source — *the second lock-free channel pattern, Part 11.5.*
**The frontiers to grow into**
- Multi-user sessions, now that conversation history can live in a shared Postgres.
- Typed tool outputs (Pydantic models) beyond today's JSON strings.
- Token streaming to a surface other than the terminal.
- Multi-user session handling — *a real `thread_id` source instead of a literal.*
- Human-in-the-loop / interrupt patterns — *a natural next node type, given the graph is
  already built to add one.*
- Actually running any part of the real-time bridge on a PREEMPT_RT-patched kernel with a
  genuinely isolated core — *the one verification this project structurally cannot do on its
  current dev machine, named as exactly that rather than implied to be done.*
- ThreadSanitizer, on an environment where it can actually launch.
- Wiring a real perception source (even a webcam + ArUco, well short of a learned policy) into
  `submit_spatial_intent` in place of the LLM tool call that exercises it today.
> **How to use this part:** for each unfamiliar line, write a one-paragraph note in your own
> words, then find or write the smallest possible experiment that proves it. That's the same
> loop that built the rest of this project — writing the fix is how you find out whether you
> actually understood the bug. Part 11 is that experiment, already run, for everything above
> that isn't Python.
---
## Part 11 — The real-time bridge, end to end
> Part 5's counterpart for `cpp/` and `spatial_tools.py`. Same standard as everywhere else in
> this document: every claim below is either backed by a specific file/line, or explicitly
> marked as not yet verified — see Part 4 and 11.6 for exactly which is which.
### 11.1 — The mental model: two processes, two clocks, no shared runtime
There is no framework here binding the two halves together — no RPC layer, no message broker, no
serialization format. There are two operating-system processes that never call into each other,
each mapping the *same* named region of physical memory into their own, independently-addressed
virtual memory, and treating a fixed byte layout inside that region as a lock-free queue. Python
runs at whatever irregular cadence the LLM decides to call a tool; the C++ side runs at a fixed
1 kHz forever, indifferent to whether anything on the other end is alive. See Part 7.5 for the
two clocks laid out side by side. The closest analogy inside this same repo is `SqliteSaver`
(Part 5.4) — a durable, out-of-process store two independent things read and write — except here
the "store" is a lock-free ring buffer in RAM instead of a file on disk, and the two readers/
writers are racing each other on purpose, at nanosecond granularity, rather than taking turns.
### 11.2 — The lock-free algorithm, in full
```cpp
bool push(const T& item) noexcept {
    const std::size_t tail = tail_.load(std::memory_order_relaxed);      // (1)
    const std::size_t next_tail = (tail + 1) & kMask;                    // (2)
    if (next_tail == head_.load(std::memory_order_acquire)) {            // (3)
        return false;  // full
    }
    buffer_[tail] = item;                                                 // (4)
    tail_.store(next_tail, std::memory_order_release);                    // (5)
    return true;
}
```
1. **Relaxed load of `tail_`.** Safe because `tail_` is *producer-owned* — no other thread ever
   writes it, so there's nothing for this thread to synchronize with when reading its own,
   only-ever-self-written value.
2. **Index wraparound via bitwise AND**, not modulo — valid only because `Capacity` is enforced
   to be a power of two (`static_assert`), which turns `(tail + 1) % Capacity` into the
   single-instruction `(tail + 1) & (Capacity - 1)`.
3. **Acquire-load of `head_`**, the *consumer's* index. This is the one load in `push()` that has
   to synchronize with the other thread: if this observes that the consumer has advanced `head_`
   past a given slot, it must also observe every memory effect the consumer performed *before*
   advancing it — otherwise the producer could overwrite a slot the consumer is still mid-read on.
4. **The payload write is plain, not atomic.** This is correct, not an oversight — see Part 6.11:
   the release/acquire pair around it is what makes this write visible to the consumer, atomicity
   on the write itself was never the mechanism doing the work.
5. **Release-store of the new `tail_`.** This is the publish step: everything written before this
   line in program order (specifically, step 4) is now guaranteed visible to any thread whose
   acquire-load of `tail_` observes this exact value.
`pop()` is the exact mirror — relaxed-load its own `head_`, acquire-load the *producer's* `tail_`
to check for new data, read the payload, release-store the advanced `head_`. Two threads, four
atomic operations total across a full push/pop cycle, zero locks, zero syscalls.
### 11.3 — Cache-line placement and the compile-time layout contract
`head_`, `tail_`, and the payload array `buffer_` each carry `alignas(kCacheLineSize)` (a fixed
64, not `std::hardware_destructive_interference_size` — GCC explicitly warns that value can
change across compiler versions or `-mtune` flags, which matters here because this *is* an ABI
boundary another language reads, not an internal implementation detail free to shift). The
resulting offsets (`head_` at 0, `tail_` at 64, `buffer_` at 128) aren't just documented, they're
`static_assert`-checked against `offsetof` on a real instantiation — Part 6.13 covers exactly why
that check had to be written as a member *function*, not inline in the class body, after it first
failed to compile. `bridge_c_api.cpp` separately `static_assert`s that the C-ABI struct
(`AgriosPose6D`) and the C++ struct (`agrios::Pose6D`) agree on *every field's* `offsetof`, not
just their combined `sizeof` — the distinction that mattered in war story #14, where a
size-only check would have missed a hypothetical field-order divergence that a per-field check
can't.
### 11.4 — POSIX shared memory: the owner/non-owner lifecycle
```
Owner (exactly one process, whichever starts first):
  shm_unlink(name)        -- clear a stale segment from a crashed prior run; ENOENT is fine
  shm_open(name, O_CREAT) -- create the named object
  ftruncate(fd, sizeof(Buffer))
  mmap(..., MAP_SHARED | MAP_POPULATE)
  memset(addr, 0, sizeof(Buffer))   -- write-touch every page; owner, so safe to zero
  new (addr) Buffer()               -- placement-construct head_=0, tail_=0

Non-owner (every other attaching process):
  shm_open(name, O_RDWR)  -- attach to the existing object, no O_CREAT
  mmap(..., MAP_SHARED | MAP_POPULATE)
  read-touch every page   -- establish this process's own page-table entries;
                              must NOT write -- would corrupt live queue state
  reinterpret_cast<Buffer*>(addr)   -- reuse the object the owner already constructed
```
Two distinct resources are in play, and conflating them is exactly the mistake war story #13
was built on: the **mapping** (`mmap`'s return value) is per-process and the kernel reclaims it
unconditionally on exit, no matter how the process dies; the **named object**
(`/dev/shm/<name>`) is not per-process, and persists until something explicitly calls
`shm_unlink` on it — see Part 6.15. Only the owner ever calls `shm_unlink`, in its destructor
*and* defensively before creating (to clear a segment orphaned by an earlier crash) — a
non-owner calling it would delete a name a still-running producer might depend on, which is
precisely why the audit in war story #13 fixed the *signal handling* that was skipping the
owner's own `shm_unlink`, not added one to the Python side.
### 11.5 — Real-time thread hardening, and what each piece actually buys
`configure_current_thread_realtime()` runs these steps, in this order, as the literal first
thing the RT thread's body does:
1. **`mlockall(MCL_CURRENT | MCL_FUTURE)`** — no page of this process, current or future, can be
   swapped out once this succeeds. It does *not* guarantee a page is *populated* before its first
   touch (Part 6.15's mapping-vs-population distinction again, in a different guise) — which is
   exactly why step 2 exists.
2. **Explicit stack pre-fault** — a large stack-local buffer is written to once, forcing the
   pages behind it to be resident before the hot loop can hit an un-faulted page from an
   unusually deep call. Requires the thread to have been created with an explicitly oversized
   stack in the first place (`make_realtime_thread_attr`), not the platform default — a default
   stack size sized for "normal" code isn't guaranteed to have room for this pre-fault pass on
   top of the loop's own usage.
3. **`SCHED_FIFO`, priority 99** — fixed-priority, no time-slicing; this thread runs until it
   blocks or something equal-or-higher-priority preempts it, unlike the default fair scheduler's
   time-sliced sharing. Priority 99 is worth knowing, not just using: it's the same priority
   Linux gives several of its own kernel housekeeping threads, so a misbehaving thread at 99 can
   starve kernel-internal work — real deployments commonly reserve 99 and run application code at
   90–98. This code uses 99 because that was the spec, which is worth distinguishing from "because
   it's the right default."
4. **`pthread_setaffinity_np()`** pinned to a chosen core — and pinning is *not* isolation
   (Part 6.16). Without also booting the kernel with `isolcpus=`/`nohz_full=`/`rcu_nocbs=` for
   that core, the scheduler can still place other work there between this thread's slices. That
   configuration is a host-kernel-boot-parameter prerequisite this code cannot set up or verify
   from inside one process.
Every one of those four calls can legitimately fail on an unprivileged process (`CAP_SYS_NICE`/
`CAP_IPC_LOCK` or root, neither granted to a default Docker container even running as root) —
`configure_current_thread_realtime` returns a hard failure with the specific reason rather than a
best-effort bool, and `motor_control_rt_loop.cpp` exits rather than proceeding when it fails. A
control loop that silently continues at normal scheduling after asking for real-time guarantees
and not getting them would fail exactly when the load it was hardened for finally showed up —
verified directly in war story-adjacent testing (Part 11.6) by removing the capabilities and
confirming the refusal, not assumed from the code.
Inside the loop itself: `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, nullptr)`
against a deadline advanced by exactly one period each iteration, not a relative `sleep_for` —
a relative sleep re-measured from "now" bakes that iteration's own execution time into the next
wait, so drift accumulates without bound over a long run; an absolute, monotonically-advancing
deadline doesn't. Telemetry (per-iteration jitter, whether a pose arrived) exits the hot loop
through a *second*, in-process instantiation of the exact same `SPSCRingBuffer<T,Capacity>`
already proven correct in 11.2/11.6 — reused for a thread-to-thread pairing this time instead of
a cross-process one — drained by a separate, ordinary-priority monitor thread that owns all the
actual `std::cout` calls the RT thread itself is forbidden from making. This is the same shape as
ROS 2 control's `RealtimeBuffer`/`RealtimePublisher`: get state out of a real-time loop without
letting a logger's I/O jitter leak back into the loop's own timing.
### 11.6 — Verification: what was actually run, and the explicit line where it stops
- **The algorithm.** A real producer thread against a real consumer thread, 5,000,000 items
  through a deliberately small 256-slot buffer forcing constant wraparound — zero lost,
  duplicated, or reordered items, clean under `-Wall -Wextra -Werror`. `ThreadSanitizer` was
  attempted for real happens-before verification beyond a stress test's lucky pass, and crashed
  on launch in this specific Docker-on-WSL2 setup regardless of the two standard workarounds
  tried — read as an environment limitation given the stress test's own result, but that read is
  a hypothesis, stated as one, not upgraded to a fact because it's convenient.
- **The bridge, cross-process and cross-language, live.** A real `motor_control_rt_loop` process
  running; a *separate* Python process importing the real `spatial_tools` module and calling the
  real `@tool`-decorated functions, not a mock — `submit_spatial_intent` delivered a pose,
  independently confirmed by the RT loop's own telemetry log switching from `pose=no` to
  `pose=yes` at the exact iteration the push landed; `read_joint_states` then read a live sample
  back from that same process.
- **The real-time mechanisms, both directions.** `mlockall`/`SCHED_FIFO(99)`/affinity all
  succeeding when the right capabilities were granted, *and* the loop correctly refusing to start
  and reporting why when they weren't — tested by actually removing the capabilities, not assumed.
- **Jitter numbers: measured, and explicitly not a real-time performance claim.** 76 μs average,
  up to several hundred microseconds to low milliseconds worst-case depending on the run — all of
  it measured inside Docker Desktop on WSL2, a stock non-PREEMPT_RT kernel, inside a VM, with no
  `isolcpus` core isolation anywhere in that stack. What's proven: the mechanisms work once
  granted privileges, and the absolute-deadline timing logic doesn't drift over a sustained run.
  What's not proven, and isn't claimed: any actual latency bound — that needs real PREEMPT_RT
  hardware and a genuinely isolated core to mean anything (Part 4).
- **Two adversarial audits, not just optimistic re-verification.** War stories #13 and #14 in
  full — two real bugs found and fixed (a `SIGTERM`-handling gap that had been masked by every
  prior ephemeral-container test run; a missing `MAP_POPULATE`/prefault pass), two demanded
  "fixes" correctly declined with technical reasoning (`#pragma pack`, a busy-spin conversion),
  and one measurement (a page-fault comparison) that came back showing the opposite of a lazy
  read's expectation and was reported exactly as measured, not reframed.
### 11.7 — The LangGraph integration layer: where the two systems actually meet
`spatial_tools.py` is the one file that imports from both worlds — `langchain_core.tools` and
`pydantic` on one side, `cpp/python/shared_ring_buffer.py`'s `ctypes` wrapper on the other — and
it's deliberately kept out of `main.py`, because its dependency footprint (a compiled `.so`,
POSIX shared memory) is fundamentally different from every other tool's (SQLAlchemy, string
fixtures). Three things make this integration layer safe rather than merely functional:
- **Validation before any C++ code runs at all.** `SubmitSpatialIntent`'s Pydantic model types
  `tvec`/`rvec` as `tuple[float, float, float]`, not `list[float]` — Pydantic v2 validates tuple
  arity exactly, so the generated JSON schema (`prefixItems`, `minItems`/`maxItems` both 3)
  rejects a malformed call from the model before `ctypes` ever marshals a byte.
- **Lazy connections, so a missing C++ side degrades instead of crashing.** `_pose_bridge`/
  `_joint_bridge` are module-level, initialized to `None`, and only opened on first tool
  invocation — `shared_ring_buffer.py`'s `_load_library()` isn't called at import time, so
  importing `spatial_tools` (and therefore `main.py`) never requires `libagrios_bridge.so` to
  exist. On this project's own Windows dev machine, where POSIX shared memory structurally can't
  exist, both tools import fine and simply report "bridge not available" as a plain string result
  when actually called — verified directly, not assumed from the lazy-import pattern being a good
  idea in general.
- **A result that distinguishes "delivered" from "acted on."** `push()` returning `true` only
  means there was room in the ring buffer — it says nothing about whether anything is actually
  reading it. `submit_spatial_intent`'s return string is explicit about which of those happened,
  because silently accepting a spatial command that will never be executed is exactly the kind of
  failure that should be surfaced to whatever's calling the tool, not swallowed into a generic
  "success."
