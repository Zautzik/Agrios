#pragma once

// The work the 1 kHz control loop does once per period, factored out of
// motor_control_rt_loop.cpp so the exact same function can be driven by the
// real-time binary and by tests/rt_safety_test.cpp. One definition, so what
// the safety checks verify is what the control loop actually runs.
//
// control_tick() is AGRIOS_NONBLOCKING (see nonblocking.h). That makes
// "no allocation, no locks, no I/O, no blocking calls in the hot path" a
// property the toolchain checks, at compile time and under
// RealtimeSanitizer, rather than a claim in a comment. The loop around it is
// deliberately not annotated: it blocks once per period in clock_nanosleep()
// by design -- waiting for the next deadline is the scheduler's job, not
// work that has to fit inside the period.

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "agrios/bridge_c_api.h"
#include "agrios/nonblocking.h"
#include "agrios/spsc_ring_buffer.hpp"

namespace agrios::control {

inline constexpr std::int64_t kPeriodNs = 1'000'000;         // 1 kHz
inline constexpr std::int64_t kOverrunThresholdNs = 200'000;  // 200 us of lateness counts as a miss

struct TelemetrySample {
    std::uint64_t iteration;
    std::int64_t jitter_ns;  // actual wake time minus the requested deadline
    bool overrun;
    bool pose_received;
    AgriosPose6D pose;  // meaningful only if pose_received
};
static_assert(std::is_trivially_copyable_v<TelemetrySample>);

// In-process only (not shared memory) -- one RT producer, one ordinary-
// priority consumer (the monitor thread). Reusing SPSCRingBuffer here is
// exactly the point: it's already proven correct (see
// tests/spsc_ring_buffer_test.cpp) for the shared-memory pose bridge, and the
// algorithm doesn't care whether the two sides are two processes or two
// threads in one process. 4096 slots is ~4 s of samples at 1 kHz: enough
// headroom that an ordinary-priority monitor thread starved by heavy load
// still doesn't lose samples, which matters because the latency report is
// built from them (and counts any that are lost).
using TelemetryRing = SPSCRingBuffer<TelemetrySample, 4096>;

// State carried from one tick to the next. Plain values only.
struct TickState {
    std::uint64_t iteration = 0;
    AgriosPose6D last_pose{};
    bool have_pose = false;
    std::uint64_t poses_received = 0;
    std::uint64_t telemetry_dropped = 0;  // samples lost because the monitor fell behind
};

// The channels a tick reads from and writes to. All pre-opened by the caller.
struct TickIo {
    AgriosBridgeHandle* pose_bridge = nullptr;
    AgriosJointBridgeHandle* joint_bridge = nullptr;
    TelemetryRing* telemetry = nullptr;
};

// One control period's worth of work. wake_latency_ns is how late this tick
// woke relative to its deadline; now_ns is the monotonic time it woke at.
inline void control_tick(TickState& state, const TickIo& io, std::int64_t wake_latency_ns,
                         std::int64_t now_ns) noexcept AGRIOS_NONBLOCKING {
    // Non-blocking, allocation-free pose read -- see
    // tests/spsc_ring_buffer_test.cpp and NOTES.md for how this was verified
    // correct under real concurrent load.
    AgriosPose6D pose{};
    if (agrios_bridge_pop(io.pose_bridge, &pose)) {
        state.last_pose = pose;
        state.have_pose = true;
        ++state.poses_received;
    }

    // Stand-in for real control-law math (PID / trajectory tracking /
    // whatever this project's motor hardware eventually needs): a fixed,
    // bounded amount of work operating only on stack values, same as the
    // real thing would need to be to stay allocation-free here.
    volatile double control_output =
        state.have_pose ? (state.last_pose.tvec[0] + state.last_pose.tvec[1] + state.last_pose.tvec[2])
                        : 0.0;
    (void)control_output;

    // Telemetry out: push() is O(1), wait-free, and never allocates or
    // blocks -- if the monitor thread is behind, this sample is dropped (and
    // counted) rather than the control loop waiting on it. Losing a
    // telemetry sample is fine; missing a control deadline to protect one
    // would not be.
    const TelemetrySample sample{state.iteration, wake_latency_ns,
                                 wake_latency_ns > kOverrunThresholdNs, state.have_pose,
                                 state.last_pose};
    if (!io.telemetry->push(sample)) {
        ++state.telemetry_dropped;
    }

    // Synthetic joint-angle telemetry, published to a separate reader process
    // over shared memory. There's no real hardware behind this yet, so the
    // "joint angles" are a clearly-synthetic waveform -- a stand-in for real
    // encoder feedback. What's real is the non-blocking push() it exercises.
    // std::sin is bounded-time and allocation-free (and Clang's
    // function-effect analysis agrees -- it would reject it otherwise).
    const double phase = static_cast<double>(state.iteration) * 0.01;
    AgriosJointState6 joint_state{};
    for (int j = 0; j < 6; ++j) {
        joint_state.joint_angles_rad[j] = 0.3 * std::sin(phase + j * 0.5);
    }
    joint_state.timestamp_ns = now_ns;
    agrios_joint_bridge_push(io.joint_bridge, &joint_state);

    ++state.iteration;
}

}  // namespace agrios::control
