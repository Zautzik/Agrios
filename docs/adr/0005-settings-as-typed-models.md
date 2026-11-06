# 0005. Settings as typed models; forms generated from them

- Status: Accepted
- Date: 2026-11-04 (recorded 2026-11-05)

## Context

"Everything adjustable through the UI" invites drift. A form can allow what the server then rejects, or the reverse, and a bad saved value can break the app that would fix it.

## Decision

- **One model per group.** Each settings group is a Pydantic model whose fields carry their limits, default, help text and whether a change applies at once or after a restart.
- **Forms come from the models.** Each field's limits, default and help text on the form are the ones the server validates.
- **Every change is logged:** who, when, old value, new value.
- **Bad stored values fall back.** A stored value that no longer validates returns to its default on its own, and an unreadable group returns to all defaults, so the page to fix it always loads.
- **Some limits are code, not settings.** Brute-force limits and the robot-tool rule are deliberately not settings: an option to weaken them would only ever be used for that.

## Consequences

- A new setting is one field in a model, and its form, validation and history come with it.
- Setting text is printed escaped, never through gettext, which marks its argument as safe HTML. A review finding.

## Evidence

`agrios/console/tests/test_settings.py`:
- form limits match the model;
- an invalid save changes nothing;
- unreadable values fall back to defaults;
- `<name>` in help text is escaped.
