// A PREEMPT_RT / ROS 2 real-time-guideline-compliant 1 kHz motor-control
// loop. Reads target poses from the same shared-memory bridge as
// motor_control_consumer.cpp (see that file for the simple, non-real-time
// version used to verify the bridge itself); this file is about the loop's
// *timing and scheduling* discipline, not new bridge functionality.
//
// Real-time requirements this satisfies, and how:
//   - mlockall(MCL_CURRENT | MCL_FUTURE) + stack pre-fault, first thing in
//     the RT thread's body -- see agrios::rt::configure_current_thread_realtime.
//   - SCHED_FIFO, priority 99 -- see the same function. (See cpp/README.md
//     for a real warning about actually using priority 99.)
//   - pthread_setaffinity_np() pinning to an isolated core -- same function;
//     see cpp/README.md for the isolcpus/nohz_full/rcu_nocbs prerequisite
//     this code cannot itself set up or verify.
//   - Zero dynamic allocation, and no std::cout, inside while(true): every
//     value the hot loop touches is a stack value or a pre-allocated buffer
//     (the shared-memory ring buffer, the telemetry ring buffer below); the
//     hot loop's only "output" is a non-blocking push() into an in-process
//     lock-free ring buffer, drained by a separate, ordinary-priority
//     monitor thread that does the actual printing. This is the same
//     pattern ros2_control's RealtimeBuffer/RealtimePublisher classes use to
//     get state out of an RT update() loop without letting a logger's I/O
//     jitter leak into the control loop's timing.
//
// Also publishes synthetic joint-angle telemetry each iteration over the
// separate joint-telemetry shared-memory channel (see joint_state.hpp),
// consumed by e.g. a ReadJointStates tool on the perception/monitoring side.
// There's no real hardware behind this yet, so the "joint angles" are a
// fixed, clearly-synthetic waveform -- a stand-in for real encoder feedback,
// same honesty as the control_output stand-in below. What's real is the
// non-blocking push() pattern it's exercising, not the numbers.
//
// The per-period work itself lives in agrios/control_tick.hpp, as an
// AGRIOS_NONBLOCKING function: the "no allocation, no locks, no I/O" rule
// above is checked by the compiler and by RealtimeSanitizer in CI, not just
// stated here. See tests/rt_safety_test.cpp.
//
// Usage: motor_control_rt_loop [--core N] [--priority P] [--iterations N]
//                              [--quiet] [--report results.json]
//   --report writes the run's measurements as JSON: wake-up latency
//   percentiles, overruns, lost telemetry samples, page faults taken by the
//   RT thread during the loop, and the kernel it ran on. tools/rt_validate.sh
//   and CI compare it against cyclictest run on the same machine.
// Requires CAP_SYS_NICE and CAP_IPC_LOCK (or root) -- e.g. under Docker:
//   docker run --cap-add=SYS_NICE --cap-add=IPC_LOCK ...

#include <pthread.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <time.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "agrios/bridge_c_api.h"
#include "agrios/bridge_config.hpp"
#include "agrios/control_tick.hpp"
#include "agrios/rt_thread.hpp"

namespace {

using agrios::control::kOverrunThresholdNs;
using agrios::control::kPeriodNs;
using agrios::control::TelemetryRing;
using agrios::control::TelemetrySample;

std::atomic<bool> g_running{true};
std::atomic<bool> g_rt_thread_failed{false};

// SIGINT and SIGTERM both need to route through here, not just SIGINT --
// SIGTERM is what `kill` sends by default, and what every real process
// supervisor (systemd, Docker, k8s) sends on shutdown before escalating to
// SIGKILL. Handling only SIGINT means every non-interactive shutdown path
// bypasses cleanup and orphans this process's two owned shared-memory
// segments. std::atomic<bool>::store with a lock-free atomic is one of the
// few operations the C++ standard actually permits inside a signal handler
// ([support.signal]) -- nothing else happens here on purpose.
static_assert(std::atomic<bool>::is_always_lock_free,
              "g_running must be lock-free to be touched from a signal handler");
void handle_shutdown_signal(int) { g_running.store(false, std::memory_order_relaxed); }

void add_ns(timespec& t, std::int64_t ns) {
    constexpr std::int64_t kNsPerSec = 1'000'000'000;
    t.tv_nsec += ns;
    while (t.tv_nsec >= kNsPerSec) {
        t.tv_nsec -= kNsPerSec;
        t.tv_sec += 1;
    }
}

std::int64_t diff_ns(const timespec& a, const timespec& b) {
    return (static_cast<std::int64_t>(a.tv_sec) - b.tv_sec) * 1'000'000'000 +
           (static_cast<std::int64_t>(a.tv_nsec) - b.tv_nsec);
}

// What the RT thread reports back once its loop ends. Written by the RT
// thread, read by main() only after pthread_join() -- the join is the
// synchronization point, so these need no atomics.
struct RtLoopResult {
    std::uint64_t iterations = 0;
    std::uint64_t poses_received = 0;
    std::uint64_t telemetry_dropped = 0;
    bool page_faults_measured = false;
    long minor_page_faults = 0;  // faults this thread took inside the loop
    long major_page_faults = 0;
};

struct RtLoopArgs {
    agrios::rt::RtThreadConfig rt_config;
    AgriosBridgeHandle* bridge = nullptr;
    AgriosJointBridgeHandle* joint_bridge = nullptr;
    TelemetryRing* telemetry = nullptr;
    std::optional<std::uint64_t> max_iterations;
    RtLoopResult result;
};

// Page faults taken by the calling thread so far. RUSAGE_THREAD is
// Linux-specific -- the same platform the rest of this file's RT setup
// (pthread_setaffinity_np) already requires. Called only before and after
// the hot loop, never inside it: getrusage() is a system call.
bool thread_page_faults(long& minor, long& major) {
#ifdef RUSAGE_THREAD
    rusage usage{};
    if (getrusage(RUSAGE_THREAD, &usage) == 0) {
        minor = usage.ru_minflt;
        major = usage.ru_majflt;
        return true;
    }
#endif
    (void)minor;
    (void)major;
    return false;
}

// The RT thread's entry point. Everything from configure_current_thread_
// realtime() onward down to the loop is the real-time-critical section;
// nothing past this function's setup prologue allocates, locks, does I/O,
// or can throw -- and the per-period work (control_tick) is checked to stay
// that way by the compiler and by RealtimeSanitizer.
void* rt_thread_main(void* arg_ptr) {
    auto& args = *static_cast<RtLoopArgs*>(arg_ptr);

    // Requirement 1-3: mlockall + stack pre-fault, SCHED_FIFO, CPU affinity.
    // "At the very beginning of the thread lifecycle" -- this is the first
    // thing this thread does, full stop.
    const agrios::rt::RtSetupResult setup = agrios::rt::configure_current_thread_realtime(args.rt_config);
    if (!setup.ok) {
        // Setup, not the hot loop -- std::cerr here is fine. Failing loudly
        // and refusing to run un-real-time is the correct behavior for code
        // that specifically exists to make real-time guarantees; see the
        // comment on configure_current_thread_realtime().
        std::cerr << "rt_thread_main: real-time setup failed: " << setup.error << '\n';
        g_rt_thread_failed.store(true, std::memory_order_relaxed);
        g_running.store(false, std::memory_order_relaxed);
        return nullptr;
    }

    agrios::control::TickState state;
    const agrios::control::TickIo io{args.bridge, args.joint_bridge, args.telemetry};

    // Baseline for the page-fault count: every page this loop touches should
    // already be resident (mlockall, the stack pre-fault, the shared-memory
    // pre-touch in SharedRingBuffer), so the loop itself should take none.
    // That's measured here, not assumed -- the delta goes into the report.
    //
    // What mlockall() doesn't cover: the kernel-provided vDSO mapping that
    // clock_gettime() runs from without a system call is set up lazily, on
    // first use. Before this warm-up read existed, that first use was the
    // loop's first clock read, and every run took exactly one minor page
    // fault inside the measured loop -- found by this measurement, not by
    // reading the code. Reading the clock once here, during setup, moves
    // that fault out of the loop (re-measured: zero faults in the loop).
    timespec deadline{};
    clock_gettime(CLOCK_MONOTONIC, &deadline);

    long minor_before = 0;
    long major_before = 0;
    const bool faults_ok = thread_page_faults(minor_before, major_before);

    clock_gettime(CLOCK_MONOTONIC, &deadline);

    // ---- Hot loop: no allocation, no std::cout, no locks, no exceptions. ----
    while (g_running.load(std::memory_order_relaxed)) {
        add_ns(deadline, kPeriodNs);

        // Absolute-deadline sleep, not a relative sleep_for(period): a
        // relative sleep re-measured from "now" each iteration accumulates
        // this iteration's own execution time as drift on every single pass
        // -- over a long-running loop that drift is unbounded. Sleeping to a
        // deadline that advances by a fixed period regardless of how long
        // the previous iteration took is what actually keeps this at 1 kHz
        // rather than "a bit under 1 kHz, worsening over time." This is the
        // loop's one intentional blocking call.
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, nullptr);

        timespec now{};
        clock_gettime(CLOCK_MONOTONIC, &now);
        const std::int64_t latency_ns = diff_ns(now, deadline);
        const std::int64_t now_ns = static_cast<std::int64_t>(now.tv_sec) * 1'000'000'000LL + now.tv_nsec;

        agrios::control::control_tick(state, io, latency_ns, now_ns);

        if (args.max_iterations.has_value() && state.iteration >= *args.max_iterations) {
            break;
        }
    }
    // ---- End hot loop. ----

    long minor_after = 0;
    long major_after = 0;
    if (faults_ok && thread_page_faults(minor_after, major_after)) {
        args.result.page_faults_measured = true;
        args.result.minor_page_faults = minor_after - minor_before;
        args.result.major_page_faults = major_after - major_before;
    }
    args.result.iterations = state.iteration;
    args.result.poses_received = state.poses_received;
    args.result.telemetry_dropped = state.telemetry_dropped;

    g_running.store(false, std::memory_order_relaxed);
    return nullptr;
}

// Wake-up latency distribution, built by the monitor thread from the
// telemetry samples (never by the RT thread). 1 us bins up to 10 ms, the same
// integer-microsecond resolution cyclictest reports, so the two are directly
// comparable; anything later lands in an overflow bin, with the exact maximum
// still tracked.
class LatencyHistogram {
   public:
    static constexpr std::size_t kBins = 10'000;

    void add(std::int64_t latency_ns) {
        ++count_;
        min_ns_ = std::min(min_ns_, latency_ns);
        max_ns_ = std::max(max_ns_, latency_ns);
        sum_ns_ += static_cast<long double>(latency_ns);
        if (latency_ns > kOverrunThresholdNs) {
            ++overruns_;
        }
        if (latency_ns >= kPeriodNs) {
            ++missed_periods_;
        }
        const std::int64_t us = std::max<std::int64_t>(latency_ns, 0) / 1000;
        if (us < static_cast<std::int64_t>(kBins)) {
            ++bins_[static_cast<std::size_t>(us)];
        } else {
            ++overflow_;
        }
    }

    // q in (0, 1]: the smallest whole microsecond value that at least q of
    // the samples fall at or below. Exact maximum if it lands in overflow.
    double percentile_us(double q) const {
        if (count_ == 0) {
            return 0.0;
        }
        auto target = static_cast<std::uint64_t>(q * static_cast<double>(count_));
        if (static_cast<double>(target) < q * static_cast<double>(count_)) {
            ++target;  // ceil
        }
        target = std::max<std::uint64_t>(target, 1);
        std::uint64_t cumulative = 0;
        for (std::size_t us = 0; us < kBins; ++us) {
            cumulative += bins_[us];
            if (cumulative >= target) {
                return static_cast<double>(us);
            }
        }
        return max_us();
    }

    std::uint64_t count() const { return count_; }
    std::uint64_t overruns() const { return overruns_; }
    std::uint64_t missed_periods() const { return missed_periods_; }
    double min_us() const { return count_ ? static_cast<double>(min_ns_) / 1000.0 : 0.0; }
    double max_us() const { return count_ ? static_cast<double>(max_ns_) / 1000.0 : 0.0; }
    double mean_us() const {
        return count_ ? static_cast<double>(sum_ns_ / static_cast<long double>(count_)) / 1000.0 : 0.0;
    }

   private:
    std::array<std::uint64_t, kBins> bins_{};
    std::uint64_t overflow_ = 0;
    std::uint64_t count_ = 0;
    std::uint64_t overruns_ = 0;
    std::uint64_t missed_periods_ = 0;
    std::int64_t min_ns_ = std::numeric_limits<std::int64_t>::max();
    std::int64_t max_ns_ = std::numeric_limits<std::int64_t>::min();
    long double sum_ns_ = 0;
};

// Ordinary SCHED_OTHER priority. Drains telemetry and does all the I/O the
// RT thread above is not allowed to do. Its own timing is not guaranteed and
// doesn't need to be -- if it falls behind, the ring buffer just fills and
// the RT thread's push()es (which never block) start being dropped, and
// counted, rather than delaying the loop.
void monitor_thread_main(TelemetryRing& telemetry, LatencyHistogram& histogram, bool quiet) {
    TelemetrySample sample{};
    while (g_running.load(std::memory_order_relaxed) || !telemetry.empty()) {
        if (!telemetry.pop(sample)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        histogram.add(sample.jitter_ns);
        if (!quiet && sample.iteration % 200 == 0) {
            std::cout << "[monitor] iter=" << sample.iteration << " jitter=" << sample.jitter_ns
                      << "ns pose=" << (sample.pose_received ? "yes" : "no") << '\n';
        }
    }
}

struct KernelInfo {
    std::string release;
    std::string version;
    std::string machine;
    bool preempt_rt = false;
};

// PREEMPT_RT kernels expose /sys/kernel/realtime = 1; their version string
// also says PREEMPT_RT. Either is enough.
KernelInfo read_kernel_info() {
    KernelInfo info;
    utsname u{};
    if (uname(&u) == 0) {
        info.release = u.release;
        info.version = u.version;
        info.machine = u.machine;
    }
    if (FILE* f = std::fopen("/sys/kernel/realtime", "r")) {
        int value = 0;
        if (std::fscanf(f, "%d", &value) == 1 && value == 1) {
            info.preempt_rt = true;
        }
        std::fclose(f);
    }
    if (info.version.find("PREEMPT_RT") != std::string::npos) {
        info.preempt_rt = true;
    }
    return info;
}

std::string json_escape(const std::string& in) {
    std::string out;
    for (const char c : in) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    out += ' ';
                } else {
                    out += c;
                }
        }
    }
    return out;
}

bool write_report(const char* path, const agrios::rt::RtThreadConfig& cfg, const RtLoopResult& result,
                  const LatencyHistogram& h, const KernelInfo& kernel, bool cpu_dma_latency_held) {
    FILE* f = std::fopen(path, "w");
    if (f == nullptr) {
        std::cerr << "could not write report to " << path << ": " << std::strerror(errno) << '\n';
        return false;
    }
    std::fprintf(f, "{\n");
    std::fprintf(f, "  \"tool\": \"agrios motor_control_rt_loop\",\n");
    std::fprintf(f, "  \"period_us\": %lld,\n", static_cast<long long>(kPeriodNs / 1000));
    std::fprintf(f, "  \"priority\": %d,\n", cfg.priority);
    std::fprintf(f, "  \"core\": %d,\n", cfg.cpu_core);
    std::fprintf(f, "  \"cpu_dma_latency_held\": %s,\n", cpu_dma_latency_held ? "true" : "false");
    std::fprintf(f, "  \"iterations\": %llu,\n", static_cast<unsigned long long>(result.iterations));
    std::fprintf(f, "  \"samples\": %llu,\n", static_cast<unsigned long long>(h.count()));
    std::fprintf(f, "  \"telemetry_dropped\": %llu,\n", static_cast<unsigned long long>(result.telemetry_dropped));
    std::fprintf(f, "  \"poses_received\": %llu,\n", static_cast<unsigned long long>(result.poses_received));
    std::fprintf(f, "  \"latency_us\": {\"min\": %.3f, \"mean\": %.3f, \"p50\": %.0f, \"p99\": %.0f, "
                    "\"p99_9\": %.0f, \"p99_99\": %.0f, \"max\": %.3f},\n",
                 h.min_us(), h.mean_us(), h.percentile_us(0.50), h.percentile_us(0.99),
                 h.percentile_us(0.999), h.percentile_us(0.9999), h.max_us());
    std::fprintf(f, "  \"overruns_over_%lldus\": %llu,\n", static_cast<long long>(kOverrunThresholdNs / 1000),
                 static_cast<unsigned long long>(h.overruns()));
    std::fprintf(f, "  \"missed_periods\": %llu,\n", static_cast<unsigned long long>(h.missed_periods()));
    if (result.page_faults_measured) {
        std::fprintf(f, "  \"page_faults_in_loop\": {\"minor\": %ld, \"major\": %ld},\n",
                     result.minor_page_faults, result.major_page_faults);
    } else {
        std::fprintf(f, "  \"page_faults_in_loop\": null,\n");
    }
    std::fprintf(f, "  \"kernel\": {\"release\": \"%s\", \"version\": \"%s\", \"machine\": \"%s\", "
                    "\"preempt_rt\": %s}\n",
                 json_escape(kernel.release).c_str(), json_escape(kernel.version).c_str(),
                 json_escape(kernel.machine).c_str(), kernel.preempt_rt ? "true" : "false");
    std::fprintf(f, "}\n");
    std::fclose(f);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    agrios::rt::RtThreadConfig rt_config;
    std::optional<std::uint64_t> max_iterations;
    bool quiet = false;
    const char* report_path = nullptr;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--core" && i + 1 < argc) {
            rt_config.cpu_core = std::atoi(argv[++i]);
        } else if (arg == "--priority" && i + 1 < argc) {
            rt_config.priority = std::atoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            max_iterations = static_cast<std::uint64_t>(std::atoll(argv[++i]));
        } else if (arg == "--quiet") {
            quiet = true;
        } else if (arg == "--report" && i + 1 < argc) {
            report_path = argv[++i];
        }
    }

    std::signal(SIGINT, handle_shutdown_signal);
    std::signal(SIGTERM, handle_shutdown_signal);

    AgriosBridgeHandle* bridge = agrios_bridge_open(agrios::kDefaultShmName, /*is_owner=*/1);
    if (bridge == nullptr) {
        std::cerr << "failed to open pose shared-memory bridge\n";
        return 1;
    }

    // This process is the owner of the joint-telemetry channel too: it's the
    // producer, and the one process guaranteed to be running whenever there's
    // telemetry to report. Readers (e.g. a ReadJointStates tool) attach as
    // non-owners.
    AgriosJointBridgeHandle* joint_bridge =
        agrios_joint_bridge_open(agrios::kDefaultJointTelemetryShmName, /*is_owner=*/1);
    if (joint_bridge == nullptr) {
        std::cerr << "failed to open joint-telemetry shared-memory bridge\n";
        agrios_bridge_close(bridge);
        return 1;
    }

    // Keep the CPU out of deep idle states for the whole run (see
    // CpuDmaLatencyGuard). Not fatal if unavailable -- it's reported.
    const agrios::rt::CpuDmaLatencyGuard dma_latency;
    if (!dma_latency.held()) {
        std::cerr << "note: " << dma_latency.error() << " (wake-up latency may include C-state exits)\n";
    }

    // Allocated once, before the RT thread starts (and before its mlockall).
    auto telemetry = std::make_unique<TelemetryRing>();
    auto histogram = std::make_unique<LatencyHistogram>();

    RtLoopArgs args;
    args.rt_config = rt_config;
    args.bridge = bridge;
    args.joint_bridge = joint_bridge;
    args.telemetry = telemetry.get();
    args.max_iterations = max_iterations;

    // Before any thread is created: this thread, and so everything it spawns
    // (the monitor thread), stays off the RT core. The RT thread moves itself
    // onto it during its own setup.
    if (!agrios::rt::keep_current_thread_off_core(rt_config.cpu_core)) {
        std::cerr << "note: could not move non-RT threads off core " << rt_config.cpu_core << '\n';
    }

    std::cout << "motor_control_rt_loop: starting monitor thread\n";
    std::thread monitor(monitor_thread_main, std::ref(*telemetry), std::ref(*histogram), quiet);

    std::cout << "motor_control_rt_loop: starting RT thread (priority=" << rt_config.priority
              << ", core=" << rt_config.cpu_core << ")\n";
    pthread_attr_t attr = agrios::rt::make_realtime_thread_attr(rt_config);
    pthread_t rt_thread{};
    if (pthread_create(&rt_thread, &attr, rt_thread_main, &args) != 0) {
        std::cerr << "pthread_create failed: " << std::strerror(errno) << '\n';
        g_running.store(false, std::memory_order_relaxed);
        monitor.join();
        agrios_joint_bridge_close(joint_bridge);
        agrios_bridge_close(bridge);
        return 1;
    }
    pthread_attr_destroy(&attr);

    pthread_join(rt_thread, nullptr);
    g_running.store(false, std::memory_order_relaxed);
    monitor.join();

    agrios_joint_bridge_close(joint_bridge);
    agrios_bridge_close(bridge);

    if (g_rt_thread_failed.load(std::memory_order_relaxed)) {
        return 1;
    }

    const LatencyHistogram& h = *histogram;
    const RtLoopResult& r = args.result;
    std::cout << "[monitor] done: " << h.count() << " samples (" << r.telemetry_dropped << " dropped), "
              << "wake-up latency us: min " << h.min_us() << " / p50 " << h.percentile_us(0.50) << " / p99 "
              << h.percentile_us(0.99) << " / p99.9 " << h.percentile_us(0.999) << " / max " << h.max_us()
              << "; " << h.overruns() << " overruns (>" << kOverrunThresholdNs / 1000 << "us), "
              << h.missed_periods() << " missed periods";
    if (r.page_faults_measured) {
        std::cout << "; page faults in RT loop: " << r.minor_page_faults << " minor, " << r.major_page_faults
                  << " major";
    }
    std::cout << '\n';

    if (report_path != nullptr) {
        const KernelInfo kernel = read_kernel_info();
        if (!write_report(report_path, rt_config, r, h, kernel, dma_latency.held())) {
            return 1;
        }
        std::cout << "report written to " << report_path << '\n';
    }
    return 0;
}
