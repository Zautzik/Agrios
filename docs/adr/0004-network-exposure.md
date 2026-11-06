# 0004. This machine only by default; the Host header checked

- Status: Accepted
- Date: 2026-11-04 (recorded 2026-11-05)

## Context

The console may be opened from a phone on the local network, or through a self-hosted VPN. Most installs never need either, and an exposed admin page is a common way in.

## Decision

- **Exposure is a setting:**
  - `local` (the default) listens on 127.0.0.1;
  - `lan` allows the RFC 1918, link-local and ULA ranges, listed explicitly;
  - `custom` takes an allow list.
- **Loopback always works.** You can't lock yourself out from the console machine.
- **Forwarded headers are ignored.** `X-Forwarded-For` has no effect, because there is no proxy.
- **The Host header must be expected.** It must be an IP address, `localhost`, or a name listed in Settings; anything else gets 421.

## Consequences

- A web page you visit can't use DNS rebinding to reach the console through your own browser. The review found that path open before the Host check was added.
- Reaching the console by a network name needs that name in Settings first.

## Evidence

`agrios/console/tests/test_access.py`:
- the client-address cases;
- the network guard on static files too;
- a forwarded header that must not change the client.

`test_auth.py` covers the rebinding case: a 421 for a foreign Host, and a pass for a configured one.
