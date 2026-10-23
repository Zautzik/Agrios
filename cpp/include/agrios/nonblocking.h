#ifndef AGRIOS_NONBLOCKING_H
#define AGRIOS_NONBLOCKING_H

/* AGRIOS_NONBLOCKING marks a function as real-time safe: bounded time, and
 * it never allocates or frees memory, takes a lock, does I/O, or makes a
 * blocking call -- directly, or anywhere in what it calls.
 *
 * With Clang 20 or newer this expands to [[clang::nonblocking]], which the
 * LLVM toolchain checks two independent ways:
 *   - at compile time, by Clang's function-effect analysis (-Wfunction-effects):
 *     calling anything not provably nonblocking from inside an annotated
 *     function is a diagnostic, and an error under -Werror;
 *   - at run time, by RealtimeSanitizer (-fsanitize=realtime): reaching
 *     malloc/free, a mutex, or a blocking system call while an annotated
 *     function is on the stack aborts the process with a stack trace.
 * CI runs both on x86-64 and arm64, each with a negative control that must
 * fail, so a passing run can't mean the check silently didn't run (see
 * tests/rt_safety_test.cpp and tests/nonblocking_negative_control.cpp).
 *
 * Every other compiler sees an empty macro, and outside a sanitizer build the
 * attribute doesn't change generated code.
 *
 * It's a function-type attribute, so it goes after the parameter list (and
 * after noexcept): `int f(int x) noexcept AGRIOS_NONBLOCKING;`
 */

#if defined(__cplusplus) && defined(__has_cpp_attribute)
#  if __has_cpp_attribute(clang::nonblocking)
#    define AGRIOS_NONBLOCKING [[clang::nonblocking]]
#  endif
#endif

#ifndef AGRIOS_NONBLOCKING
#  define AGRIOS_NONBLOCKING
#endif

#endif /* AGRIOS_NONBLOCKING_H */
