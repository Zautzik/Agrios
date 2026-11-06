# 0006. The browser Assistant gets the farm tools only

- Status: Accepted
- Date: 2026-11-05

## Context

The agent's tools include `submit_spatial_intent`, which sends a target pose to the robot arm's real-time loop. In the CLI, the person typing is at the machine. In a browser, the message could come from any signed-in account, on any device on the network.

## Decision

- **Farm tools only.** The console runs the same agent graph as the CLI, but with only the farm tools: forecast, crop calendar, field status, log task and pending tasks.
- **Enforced twice:**
  - Only those tools are bound to the model for a console run.
  - The tool node refuses any other call before it runs, because a model can emit a call to a tool it was never given.
- **The CLI keeps every tool.** Moving hardware from the browser waits for the Hardware screen (C3), with an explicit confirmation step that is logged.

## Consequences

Asking the browser Assistant to move the arm gets a refusal. It can still talk about the arm, but it can't act on it.

## Evidence

`agrios/console/tests/test_assistant.py`:
- The model is given exactly the farm tools.
- A scripted robot call from the console is refused and never reaches the tool.
- The negative control shows the same guard lets the call through without the console's restriction.
- Removing either layer makes its test fail.
