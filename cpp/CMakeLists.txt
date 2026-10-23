cmake_minimum_required(VERSION 3.16)
project(agrios_bridge CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin"))
  message(FATAL_ERROR
    "agrios_bridge uses POSIX shared memory (shm_open/mmap) and is only "
    "meaningful on Linux/macOS -- it targets the same Linux boxes and "
    "Raspberry Pi / Jetson boards the rest of the project runs on. Build "
    "under WSL, a Linux container, or a real Linux/macOS host.")
endif()

add_compile_options(-Wall -Wextra)

find_package(Threads REQUIRED)

# Shared library exposing the plain-C ABI Python's ctypes bridge calls into.
add_library(agrios_bridge SHARED
  src/bridge_c_api.cpp
)
target_include_directories(agrios_bridge PUBLIC include)
target_link_libraries(agrios_bridge PUBLIC rt Threads::Threads)

# Demo: the "C++ motor control" consumer -- simple, non-real-time, verifies
# the bridge itself (see cpp/README.md and NOTES.md for that verification).
add_executable(motor_control_consumer src/motor_control_consumer.cpp)
target_link_libraries(motor_control_consumer PRIVATE agrios_bridge)

# The actual PREEMPT_RT / ROS 2 real-time-guideline-compliant 1 kHz control
# loop -- see motor_control_rt_loop.cpp's top comment and cpp/README.md.
# -Wl,-z,now forces eager (load-time) symbol binding instead of lazy binding:
# without it, the *first* call to a given dynamically-linked function from
# inside the RT hot loop pays the cost of the dynamic linker resolving that
# symbol on demand, which can itself fault in pages -- exactly the kind of
# non-deterministic first-touch latency this loop exists to eliminate.
add_executable(motor_control_rt_loop src/motor_control_rt_loop.cpp)
target_link_libraries(motor_control_rt_loop PRIVATE agrios_bridge Threads::Threads)
target_link_options(motor_control_rt_loop PRIVATE -Wl,-z,now)

# Correctness test for the lock-free algorithm itself (no shared memory
# involved -- two threads in one process). Run manually with TSan for real
# signal; see the comment at the top of the test file.
add_executable(spsc_ring_buffer_test tests/spsc_ring_buffer_test.cpp)
target_include_directories(spsc_ring_buffer_test PRIVATE include)
target_link_libraries(spsc_ring_buffer_test PRIVATE Threads::Threads)

# Real-time-safety test for the control step (agrios/control_tick.hpp): run
# under RealtimeSanitizer in CI, where any allocation, lock, or blocking call
# reached from control_tick aborts it. In an ordinary build it's a functional
# test of the same function. See the comment at the top of the file.
add_executable(rt_safety_test tests/rt_safety_test.cpp)
target_link_libraries(rt_safety_test PRIVATE agrios_bridge Threads::Threads)

enable_testing()
add_test(NAME spsc_ring_buffer_test COMMAND spsc_ring_buffer_test)
add_test(NAME rt_safety_test COMMAND rt_safety_test)
