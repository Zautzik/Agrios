# 0001. Record decisions, and back them with evidence

- Status: Accepted
- Date: 2026-11-05

## Context

Agrios is built one phase at a time, by one person, and is also read by people judging the work. The reasons behind a choice are easy to lose: why there's no React, why the browser Assistant can't move the arm. A reason that only lives in a chat log is lost. A claim with no measurement behind it can't be checked.

## Decision

- **Write it down.** Significant decisions are recorded here, one file each.
- **Evidence first.** A decision that rests on a measurement links the command or test that produced it.
- **Negative controls.** Every check that guards a decision has one: a deliberately broken input it must reject. A test that can't fail proves nothing.
- **Mutation checks.** Before a phase is committed, its protections are removed one at a time, and each removal must make a test fail.
- **Independent review.** Each phase ends with a review by a reviewer with fresh context, and its confirmed findings are fixed with tests.

## Consequences

- Each phase costs more time.
- In return, claims in the README can be checked, and regressions show up in CI rather than in use.
- The build log ([NOTES.md](../../NOTES.md)) keeps the story, including what went wrong. These records keep only the decisions.

## Evidence

- **Phase 1:** the vision gate's controls must fail in CI (`--expect-fail`).
- **C1c:** 17 of 17 removed protections were caught by the tests.
- **C1d:** 4 of 4 were caught by the browser suite. One mutation that wasn't caught turned out not to be a regression: removing the focus rule leaves the browser's own visible focus ring.
