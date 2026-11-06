# 0002. Server-rendered pages with htmx, no build step

- Status: Accepted
- Date: 2026-11-03 (recorded 2026-11-05)

## Context

The console is maintained by one person whose code is otherwise Python. It runs offline on a farm or workshop machine, and must work on a phone with gloves on.

## Decision

- **Rendering:** Jinja templates rendered by FastAPI.
- **htmx 2:** used for partial updates and boosted navigation. It's vendored with its hash checked, and configured so it needs no `eval` and no inline styles.
- **Plain CSS:** hand-written stylesheet with design tokens.
- **No Node toolchain** and no CDN.

## Consequences

**What we get:**
- **One language and one process.** Pages work without JavaScript: the no-JS paths are tested.
- **A strict policy.** The Content-Security-Policy can be `script-src 'self'; style-src 'self'` with no exceptions.

**What we give up:**
- Rich client interactions take hand-written JavaScript. `assistant.js` (streaming) is the example so far, and Alpine.js arrives with the calibration canvas in C2.
- htmx's defaults needed care: its settle step copies `style` attributes, which the CSP refuses. A browser run caught it, and `attributesToSettle` was narrowed.

## Evidence

- **Browser suite:** `agrios/console/tests/e2e` loads every screen at 1280 and 390 px, with no CSP violation and no script error. A deliberately broken page with an inline script and style must trip the check.
- **Speed:** a page costs the server 5–6 ms median on the dev laptop (`scripts/console_perf.py`).
