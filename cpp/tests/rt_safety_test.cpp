// Real-time-safety test for the 1 ms control step, agrios::control::control_tick.
//
// control_tick is AGRIOS_NONBLOCKING (see agrios/nonblocking.h). Built with
// Clang 20+ and -fsanitize=realtime, that makes LLVM's RealtimeSanitizer
// abort this process the moment anything control_tick reaches -- directly or
// deep inside a library -- allocates, frees, takes a lock, or makes a
// blocking call. CI builds and runs it that way on x86-64 and arm64.
//
// It drives the production function itself (one definition, shared with
// motor_control_rt_loop), over real POSIX shared-memory bridges, with a live
// producer thread, so both the "no pose yet" and "pose received" paths
// execute, and the telemetry ring both drains and fills.
//
// --negative-control deliberately calls an allocating function from a
// nonblocking context. RealtimeSanitizer must kill it; CI checks that it
// does. A sanitizer that silently isn't active would otherwise make a
// passing run look like evidence when it's evidence of nothing.
//
// In an ordinary (non-sanitizer) build this still runs as a functional test
// of control_tick, registered with CTest.

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include "agrios/bridge_c_api.h"
#include "agrios/control_tick.hpp"
#include "agrios/nonblocking.h"

#if defined(__has_feature)
#  if __has_feature(realtime_sanitizer)
#    define AGRIOS_RTSAN_ACTIVE 1
#  endif
#endif
#ifndef AGRIOS_RTSAN_ACTIVE
#  define AGRIOS_RTSAN_ACTIVE 0
#endif

namespace {

// Same rationale as spsc_ring_buffer_test.cpp: assert() compiles to nothing
// under NDEBUG, which every CMake Release-family build defines.
#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::fprintf(stderr, "CHECK FAILED at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::abort();                                                                   \
        }                                                                                   \
    } while (0)

constexpr std::uint64_t kTicks = 20'000;

// The negative control: a nonblocking function that allocates. The static
// analysis would rightly reject this, so it's silenced for this one function
// -- the point here is to prove the *runtime* check fires.
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wunknown-warning-option"
#  pragma clang diagnostic ignored "-Wfunction-effects"
#endif
// Published through a volatile global so the allocation escapes: C++ lets
// the compiler delete an unobserved new/delete pair outright, and the first
// draft of this negative control was optimized away exactly like that -- it
// "passed" without allocating anything. Which is the failure mode this
// control exists to catch, caught on itself.
int* volatile g_negative_control_sink = nullptr;

[[gnu::noinline]] int allocate_in_realtime_context(int seed) noexcept AGRIOS_NONBLOCKING {
    g_negative_control_sink = new int(seed);  // must be caught: heap allocation in RT context
    const int out = *g_negative_control_sink;
    delete g_negative_control_sink;
    return out;
}
#if defined(__clang__)
#  pragma clang diagnostic pop
#endif

int run_negative_control() {
    if (!AGRIOS_RTSAN_ACTIVE) {
        std::fprintf(stderr,
                     "rt_safety_test: --negative-control needs a -fsanitize=realtime build; "
                     "this binary wasn't built with it\n");
        return 2;
    }
    std::printf("rt_safety_test: negative control -- allocating inside a nonblocking function; "
                "RealtimeSanitizer should abort now\n");
    std::fflush(stdout);
    const int v = allocate_in_realtime_context(42);
    // Only reached if the sanitizer did NOT fire -- which is the failure.
    std::fprintf(stderr, "rt_safety_test: negative control was NOT caught (got %d)\n", v);
    return 1;
}

int run_safety_test() {
    const std::string suffix = std::to_string(static_cast<long>(::getpid()));
    const std::string pose_name = "/agrios_rtsafety_pose_" + suffix;
    const std::string joint_name = "/agrios_rtsafety_joint_" + suffix;

    AgriosBridgeHandle* pose_bridge = agrios_bridge_open(pose_name.c_str(), /*is_owner=*/1);
    CHECK(pose_bridge != nullptr);
    AgriosJointBridgeHandle* joint_bridge = agrios_joint_bridge_open(joint_name.c_str(), /*is_owner=*/1);
    CHECK(joint_bridge != nullptr);
    auto telemetry = std::make_unique<agrios::control::TelemetryRing>();

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> poses_pushed{0};

    // Producer: a stand-in for the Python side pushing poses at its own pace.
    // Starts after a short delay, so the first ticks run with no pose yet.
    std::thread producer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        AgriosPose6D pose{};
        std::uint64_t n = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            pose.tvec[0] = 0.1 * static_cast<double>(n % 10);
            pose.timestamp_ns = static_cast<std::int64_t>(n);
            if (agrios_bridge_push(pose_bridge, &pose)) {
                ++n;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        poses_pushed.store(n, std::memory_order_relaxed);
    });

    // Telemetry consumer that only drains for the first half of the run, so
    // the ring also fills and control_tick takes its "monitor fell behind,
    // drop and count the sample" path.
    std::atomic<bool> drain{true};
    std::thread drainer([&] {
        agrios::control::TelemetrySample sample{};
        while (!stop.load(std::memory_order_relaxed)) {
            if (drain.load(std::memory_order_relaxed)) {
                while (telemetry->pop(sample)) {
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    });

    agrios::control::TickState state;
    const agrios::control::TickIo io{pose_bridge, joint_bridge, telemetry.get()};
    for (std::uint64_t i = 0; i < kTicks; ++i) {
        // Synthetic wake-up latency and timestamp -- this test is about what
        // control_tick calls, not about timing.
        agrios::control::control_tick(state, io, static_cast<std::int64_t>(i % 500) * 1000,
                                      static_cast<std::int64_t>(i) * agrios::control::kPeriodNs);
        if (i == kTicks / 2) {
            drain.store(false, std::memory_order_relaxed);
        }
        if (i % 100 == 0) {
            // Outside control_tick, so not real-time context: lets the
            // producer interleave with the ticks.
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    stop.store(true, std::memory_order_relaxed);
    producer.join();
    drainer.join();

    CHECK(state.iteration == kTicks);
    CHECK(state.have_pose);              // the pose-received path ran
    CHECK(state.poses_received > 0);
    CHECK(state.telemetry_dropped > 0);  // the ring-full path ran too

    agrios_joint_bridge_close(joint_bridge);
    agrios_bridge_close(pose_bridge);

    std::printf("rt_safety_test: %llu control ticks, %llu poses received (of %llu pushed), "
                "%llu telemetry samples dropped by design; RealtimeSanitizer %s\n",
                static_cast<unsigned long long>(state.iteration),
                static_cast<unsigned long long>(state.poses_received),
                static_cast<unsigned long long>(poses_pushed.load()),
                static_cast<unsigned long long>(state.telemetry_dropped),
                AGRIOS_RTSAN_ACTIVE ? "active: no real-time-unsafe call reached"
                                    : "not active in this build (functional run only)");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--negative-control") == 0) {
        return run_negative_control();
    }
    return run_safety_test();
}
