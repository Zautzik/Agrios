# Architecture decision records

Short records of the decisions that shape Agrios. Each one gives the problem, what was chosen and what was given up, and the evidence behind it. The format is [MADR](https://adr.github.io/madr/)'s, trimmed to four sections.

A record isn't edited once accepted. If a decision changes, a new record supersedes it and links back.

| # | Decision | Status |
|---|---|---|
| [0001](0001-record-decisions.md) | Record decisions, and back them with evidence | Accepted |
| [0002](0002-server-rendered-htmx.md) | Server-rendered pages with htmx, no build step | Accepted |
| [0003](0003-local-accounts-server-sessions.md) | Local accounts, argon2id, server-side sessions | Accepted |
| [0004](0004-network-exposure.md) | This machine only by default; the Host header checked | Accepted |
| [0005](0005-settings-as-typed-models.md) | Settings as typed models; forms generated from them | Accepted |
| [0006](0006-browser-assistant-farm-tools-only.md) | The browser Assistant gets the farm tools only | Accepted |
| [0007](0007-stream-over-post-detached-turns.md) | Stream replies over POST; a turn outlives its connection | Accepted |
| [0008](0008-licences-include-test-tools.md) | The licence rule covers test tools too | Accepted |
