# 0007. Stream replies over POST; a turn outlives its connection

- Status: Accepted
- Date: 2026-11-05

## Context

A reply from the local model takes seconds to minutes on CPU-only hardware: about 2 minutes for a full answer on the dev laptop. It has to appear as it's written, show each tool the agent uses, and survive the page being reloaded.

## Decision

- **Server-Sent Events over the POST that sends the message.** This uses FastAPI's built-in `EventSourceResponse`, read with `fetch()`.
  - EventSource can only GET, and the message and CSRF token belong in a body and a header, not a URL.
  - sse-starlette isn't needed.
- **A no-JS path.** Without JavaScript, the same form posts to `/assistant/send`, which waits and redirects back.
- **Each turn runs on its own thread**, and the response only passes its events on.
- **One turn at a time.** A conversation accepts one turn at a time, and deleting it waits for the running turn.

## Consequences

**Why the thread:** FastAPI closes a streaming generator when the client goes away. Inline, that stopped the agent between a tool call and its result, leaving a half-saved turn. On its own thread the turn finishes and is saved, and the next page load shows it.

**Costs:**
- A running turn can't be cancelled from the browser.
- The busy flag and the threads are per process, consistent with ADR 0003.

## Evidence

**Found by review:** a real server, a slow model and a closed stream left `[human, ai '']` saved.

**`test_assistant.py`:**
- a stream closed after its first event still saves `[human, ai, tool, ai]`;
- the control shows a generator consumed inline stops where it's closed;
- a delete while answering is refused;
- a turn whose conversation was deleted removes its checkpoint.

**My first control** abandoned a real turn inline. Windows then reported an access violation about one run in three: LangGraph's background write raced the checkpointer's close. That's the hazard this design removes, so the control was rewritten not to trigger it.

**Measured:** with a model that answers instantly, the console takes 48 ms median from send to the first word (`scripts/console_perf.py`).
