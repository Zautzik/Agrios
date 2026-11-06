# 0008. The licence rule covers test tools too

- Status: Accepted
- Date: 2026-11-05

## Context

Agrios is MIT, and its dependencies must be MIT, BSD or Apache-2.0. FFmpeg's LGPL, dynamically linked through OpenCV, is the one accepted exception (decision D1). The standard accessibility checker, axe-core, is MPL-2.0, and so is Hypothesis, the usual property-testing library. Lighthouse is Apache-2.0 but bundles axe-core.

## Decision

- **The rule applies to development and test tools as well as runtime ones.** So axe-core, Lighthouse and Hypothesis are not used.
- **Accessibility is checked with our own tests** on Playwright (Apache-2.0):
  - contrast, computed from the stylesheet's tokens in both themes against WCAG 2.2 AA;
  - a name for every visible form control, button and link;
  - a visible focus indicator on every keyboard stop, reached by tabbing;
  - no horizontal scroll at 390 px.
- **No property-testing library.** Property-style tests use seeded parametrisation instead.

## Consequences

- **The checks are narrower than axe's rule set.** They cover the failures this UI is most likely to have, not all of WCAG. Things they don't check: ARIA validity, heading order, landmark structure, and announcements while a reply streams. Those were reviewed by hand, and a usability test with real users is planned for C4.
- **One licence rule, no exceptions to track.** A test-only tool never ships, but a written exception has to be judged every time, and allowing MPL in tests would still be an exception.

## Evidence

- **Contrast:** `agrios/console/tests/test_contrast.py`, whose negative control lightens one token and must fail.
- **Browser checks:** `agrios/console/tests/e2e`. A broken page with an unlabelled input must fail, and `outline: none` on focus must fail the keyboard test.
