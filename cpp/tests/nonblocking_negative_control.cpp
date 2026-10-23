// Negative control for the compile-time real-time-safety check. This file
// must FAIL to compile with Clang 20+ under -Werror=function-effects; CI
// asserts that it does (it is not part of the CMake build).
//
// Why it exists: the check on control_tick (agrios/control_tick.hpp) is only
// evidence if the analysis is actually running. If AGRIOS_NONBLOCKING ever
// silently expanded to nothing -- a renamed attribute, a compiler-version
// check gone wrong -- control_tick would keep compiling cleanly and CI would
// stay green while checking nothing. This file uses the same macro and the
// same kind of violation, so that failure mode turns CI red instead.

#include <cstdlib>

#include "agrios/nonblocking.h"

int allocates_in_a_nonblocking_function() noexcept AGRIOS_NONBLOCKING {
    void* p = std::malloc(16);  // must be rejected: malloc is not nonblocking
    std::free(p);
    return 0;
}
