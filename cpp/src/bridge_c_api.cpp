#include "agrios/bridge_c_api.h"

#include <cstddef>  // offsetof
#include <cstring>
#include <exception>
#include <iostream>

#include "agrios/bridge_config.hpp"
#include "agrios/joint_state.hpp"
#include "agrios/pose6d.hpp"
#include "agrios/shared_ring_buffer.hpp"

namespace {

using PoseBridge = agrios::SharedRingBuffer<agrios::Pose6D, agrios::kBridgeCapacity>;
using JointBridge = agrios::SharedRingBuffer<agrios::JointState6, agrios::kJointTelemetryCapacity>;

// Per-field offset checks, not just sizeof: two structs can have identical
// total size while their fields have silently diverged in order -- a
// sizeof-only check would not catch that, and it's exactly the kind of
// change that's easy to make on one side of this file and forget on the
// other, since the two struct definitions live in different headers
// (bridge_c_api.h vs pose6d.hpp / joint_state.hpp).
static_assert(sizeof(AgriosPose6D) == sizeof(agrios::Pose6D) &&
                  offsetof(AgriosPose6D, tvec) == offsetof(agrios::Pose6D, tvec) &&
                  offsetof(AgriosPose6D, rvec) == offsetof(agrios::Pose6D, rvec) &&
                  offsetof(AgriosPose6D, timestamp_ns) == offsetof(agrios::Pose6D, timestamp_ns),
              "AgriosPose6D (C ABI) and agrios::Pose6D (C++ implementation) "
              "have drifted apart -- keep their fields identical in order "
              "and type");
static_assert(sizeof(AgriosJointState6) == sizeof(agrios::JointState6) &&
                  offsetof(AgriosJointState6, joint_angles_rad) ==
                      offsetof(agrios::JointState6, joint_angles_rad) &&
                  offsetof(AgriosJointState6, timestamp_ns) ==
                      offsetof(agrios::JointState6, timestamp_ns),
              "AgriosJointState6 (C ABI) and agrios::JointState6 (C++ "
              "implementation) have drifted apart -- keep their fields "
              "identical in order and type");

}  // namespace

struct AgriosBridgeHandle {
    PoseBridge bridge;
};

struct AgriosJointBridgeHandle {
    JointBridge bridge;
};

extern "C" {

/* ---- Pose bridge ---- */

AgriosBridgeHandle* agrios_bridge_open(const char* shm_name, int is_owner) {
    if (shm_name == nullptr) {
        return nullptr;
    }
    try {
        return new AgriosBridgeHandle{PoseBridge(shm_name, is_owner != 0)};
    } catch (const std::exception& e) {
        std::cerr << "agrios_bridge_open failed: " << e.what() << '\n';
        return nullptr;
    }
}

void agrios_bridge_close(AgriosBridgeHandle* handle) { delete handle; }

int agrios_bridge_push(AgriosBridgeHandle* handle, const AgriosPose6D* pose) AGRIOS_NONBLOCKING {
    if (handle == nullptr || pose == nullptr) {
        return 0;
    }
    // AgriosPose6D and agrios::Pose6D are field-for-field identical (checked
    // above); this is a same-layout reinterpretation, not a type pun across
    // unrelated types.
    const auto& typed_pose = *reinterpret_cast<const agrios::Pose6D*>(pose);
    return handle->bridge.get().push(typed_pose) ? 1 : 0;
}

int agrios_bridge_pop(AgriosBridgeHandle* handle, AgriosPose6D* out) AGRIOS_NONBLOCKING {
    if (handle == nullptr || out == nullptr) {
        return 0;
    }
    auto& typed_out = *reinterpret_cast<agrios::Pose6D*>(out);
    return handle->bridge.get().pop(typed_out) ? 1 : 0;
}

/* ---- Joint telemetry ---- */

AgriosJointBridgeHandle* agrios_joint_bridge_open(const char* shm_name, int is_owner) {
    if (shm_name == nullptr) {
        return nullptr;
    }
    try {
        return new AgriosJointBridgeHandle{JointBridge(shm_name, is_owner != 0)};
    } catch (const std::exception& e) {
        std::cerr << "agrios_joint_bridge_open failed: " << e.what() << '\n';
        return nullptr;
    }
}

void agrios_joint_bridge_close(AgriosJointBridgeHandle* handle) { delete handle; }

int agrios_joint_bridge_push(AgriosJointBridgeHandle* handle,
                             const AgriosJointState6* state) AGRIOS_NONBLOCKING {
    if (handle == nullptr || state == nullptr) {
        return 0;
    }
    const auto& typed_state = *reinterpret_cast<const agrios::JointState6*>(state);
    return handle->bridge.get().push(typed_state) ? 1 : 0;
}

int agrios_joint_bridge_pop(AgriosJointBridgeHandle* handle, AgriosJointState6* out) AGRIOS_NONBLOCKING {
    if (handle == nullptr || out == nullptr) {
        return 0;
    }
    auto& typed_out = *reinterpret_cast<agrios::JointState6*>(out);
    return handle->bridge.get().pop(typed_out) ? 1 : 0;
}

}  // extern "C"
