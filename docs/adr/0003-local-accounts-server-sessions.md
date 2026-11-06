# 0003. Local accounts, argon2id, server-side sessions

- Status: Accepted
- Date: 2026-11-04 (recorded 2026-11-05)

## Context

The console shows farm data and, later, camera footage, and it can change settings. It runs offline, so there's no identity provider to lean on.

## Decision

- **Accounts:** local, with three roles: owner, operator and viewer. An access matrix test lists every route and who may use it.
- **Passwords:** argon2id with argon2-cffi's defaults (the RFC 9106 low-memory profile). At least 15 characters and no composition rules, per NIST SP 800-63B-4.
- **Sessions:** stored server-side.
  - The cookie holds a 256-bit random token. Only its SHA-256 is stored.
  - A new token on every sign-in.
  - An idle and an absolute timeout, both settings.
- **CSRF:** a synchronizer token, plus Fetch Metadata and Origin checks, plus SameSite=Strict.
- **Brute force:** backoff per username and client address, checked under a lock.

## Consequences

- **Immediate revocation.** Signing out or deleting a user ends the session at once, and a copied database can't be used to sign in.
- **Single process.** The locks are per process, so the console runs as one uvicorn worker. More workers would need database locks.
- **Cheap session checks.** Writing the idle timer on every request cost half of every page. It's now refreshed at most once a minute, invisible against a 5-minute minimum timeout. An expired session must stay expired even if the timeout is then lengthened, so the settings page sweeps expired sessions before changing it.

## Evidence

- **Backoff:** the review found that 40 parallel wrong passwords were all checked. With the lock, 5 are checked and 35 refused, and a test repeats that experiment.
- **Session refresh:** page time on the dev laptop went from 11.07 to 5.38 ms median (p95 17.89 to 8.23 ms) after the change.
  - Tests check that a request within the minute doesn't write.
  - Tests check that a longer timeout doesn't revive a session that expired just after the last sweep.
  - Removing either protection makes its test fail.
