#ifndef AGRIOS_BRIDGE_C_API_H
#define AGRIOS_BRIDGE_C_API_H

/* Plain-C ABI over two independent shared-memory channels:
 *   - the pose bridge:      SharedRingBuffer<Pose6D, kBridgeCapacity>
 *   - the joint telemetry:  SharedRingBuffer<JointState6, kJointTelemetryCapacity>
 * See bridge_config.hpp for their names/capacities and pose6d.hpp /
 * joint_state.hpp for the payload layouts.
 *
 * Why this exists instead of letting Python poke the shared-memory bytes
 * directly: push()/pop()'s correctness depends on real acquire/release
 * memory fences, not just on reading/writing the right bytes at the right
 * offsets. Python's struct/ctypes modules have no memory-order-aware atomic
 * primitives, and a naive load/store would only happen to work by accident
 * on x86-64 (whose strong memory model papers over the missing fences) --
 * it would be a real, silent bug on a weakly-ordered target like the
 * ARM64 Jetson/Raspberry Pi boards this bridge is meant to eventually run
 * motor control on. Routing Python through this compiled shim means both
 * languages execute the exact same std::atomic operations, so there's one
 * implementation of the protocol, not two that have to be kept in sync by
 * hand.
 *
 * Two channels, two small sets of near-identical functions rather than one
 * generic void*-based API: the C ABI can't carry a template parameter, and a
 * type-erased API would trade compile-time payload-type safety for avoiding
 * ~15 lines of duplication. Not worth it for two types.
 */

#include <stdint.h>

#include "agrios/nonblocking.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Pose bridge: perception -> motor control ---- */

typedef struct AgriosPose6D {
    double tvec[3];       /* translation vector [tx, ty, tz], meters */
    double rvec[3];       /* Rodrigues rotation vector [rx, ry, rz], radians */
    int64_t timestamp_ns; /* monotonic clock, nanoseconds */
} AgriosPose6D;

typedef struct AgriosBridgeHandle AgriosBridgeHandle; /* opaque */

/* Creates or attaches to the named POSIX shared-memory segment.
 * is_owner = 1: this process creates/resets the segment (call exactly once,
 *   from whichever process starts first -- it clears any stale segment from
 *   a crashed prior run and zero-initializes the ring buffer).
 * is_owner = 0: this process attaches to an already-created segment.
 * Returns NULL on failure. */
AgriosBridgeHandle* agrios_bridge_open(const char* shm_name, int is_owner);

/* Unmaps and closes the handle. If it was opened with is_owner = 1, also
 * unlinks the shared-memory segment. */
void agrios_bridge_close(AgriosBridgeHandle* handle);

/* push/pop below are AGRIOS_NONBLOCKING (see nonblocking.h): the RT control
 * step calls them every millisecond, so they are compile-time and
 * sanitizer-checked never to allocate, lock, or block. open/close are not --
 * they allocate and make system calls, and belong in setup/teardown only. */

/* Returns 1 on success, 0 if the ring buffer is full. */
int agrios_bridge_push(AgriosBridgeHandle* handle, const AgriosPose6D* pose) AGRIOS_NONBLOCKING;

/* Returns 1 on success (out is filled in), 0 if the ring buffer is empty. */
int agrios_bridge_pop(AgriosBridgeHandle* handle, AgriosPose6D* out) AGRIOS_NONBLOCKING;

/* ---- Joint telemetry: motor control -> monitoring/perception ---- */

typedef struct AgriosJointState6 {
    double joint_angles_rad[6];
    int64_t timestamp_ns;
} AgriosJointState6;

typedef struct AgriosJointBridgeHandle AgriosJointBridgeHandle; /* opaque */

/* Same open/create semantics as agrios_bridge_open -- see above. In the
 * normal topology the motor-control RT loop is the owner (it's the producer
 * and the one process guaranteed to be running whenever there's telemetry to
 * report), and readers (e.g. a ReadJointStates tool) attach as non-owners. */
AgriosJointBridgeHandle* agrios_joint_bridge_open(const char* shm_name, int is_owner);
void agrios_joint_bridge_close(AgriosJointBridgeHandle* handle);

/* Returns 1 on success, 0 if the ring buffer is full. */
int agrios_joint_bridge_push(AgriosJointBridgeHandle* handle,
                             const AgriosJointState6* state) AGRIOS_NONBLOCKING;

/* Returns 1 on success (out is filled in), 0 if the ring buffer is empty.
 * Never blocks the writer -- SPSCRingBuffer's pop() is wait-free by
 * construction (see spsc_ring_buffer.hpp), so a reader calling this, however
 * slowly or however often, can never stall the RT loop's push()es. */
int agrios_joint_bridge_pop(AgriosJointBridgeHandle* handle, AgriosJointState6* out) AGRIOS_NONBLOCKING;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AGRIOS_BRIDGE_C_API_H */
