"""Local accounts, server-side sessions and login backoff.

References for the choices below:
- Passwords: argon2id via argon2-cffi's defaults, which are the RFC 9106
  "second recommended" (low-memory) profile. Minimum length 15 and no
  composition rules, per NIST SP 800-63B-4 section 3.1.1.2 for a password
  that is the only authenticator.
- Sessions: 256-bit random tokens; only their SHA-256 is stored; a new token
  on every sign-in (no session fixation); idle and absolute timeouts
  (OWASP Session Management Cheat Sheet).
- Backoff: per (username, client address), so someone guessing from one
  machine can't lock the owner out of every other machine.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import threading
from dataclasses import dataclass
from typing import Literal

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from sqlalchemy import delete, func, insert, or_, select, update

from agrios.db import engine, login_attempts_table, sessions_table, users_table

Role = Literal["owner", "operator", "viewer"]
ROLES: tuple[Role, ...] = ("owner", "operator", "viewer")

USERNAME_PATTERN = re.compile(r"^[a-z0-9._-]{1,32}$")
MIN_PASSWORD_LENGTH = 15
MAX_PASSWORD_LENGTH = 128  # bounds the cost of hashing a hostile input

# Fixed by design, like the safety invariants: making brute-force protection
# adjustable would only ever be used to weaken it.
FREE_ATTEMPTS = 5
FIRST_LOCK_S = 30.0
MAX_LOCK_S = 900.0
# A failure record this old is forgotten, so occasional typos never add up.
FORGET_FAILURES_AFTER_S = 24 * 3600.0

_hasher = PasswordHasher()
# Verified against when the username doesn't exist, so a wrong username takes
# as long as a wrong password and response time doesn't reveal which users exist.
_DUMMY_HASH = _hasher.hash(secrets.token_urlsafe(16))
# Both locks are per process. The console runs as a single process (one uvicorn
# worker, see __main__), which is what makes check-then-write safe here; more
# workers would need these as database row locks instead.
_setup_lock = threading.Lock()
# Serialises sign-in checks, so concurrent guesses can't all read the same
# failure count before any of them records a failure (found in review: 40
# parallel wrong passwords were all checked, none locked out). At ~50 ms per
# argon2 verification it also caps guessing at ~20 per second per process.
_login_lock = threading.Lock()


@dataclass(frozen=True)
class User:
    id: int
    username: str
    role: Role


@dataclass(frozen=True)
class Session:
    user: User
    csrf_token: str


class SetupClosedError(Exception):
    """The first-run setup was attempted after an account already exists."""


class InvalidCredentialsError(ValueError):
    """Username or password rejected by the account rules, with a message for the form."""


@dataclass(frozen=True)
class LockedOut:
    retry_after_s: float


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def normalize_username(username: str) -> str:
    return username.strip().lower()


def check_new_credentials(username: str, password: str) -> None:
    if not USERNAME_PATTERN.fullmatch(username):
        raise InvalidCredentialsError(
            "Usernames are 1 to 32 characters: lowercase letters, digits, dot, dash or underscore."
        )
    if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
        raise InvalidCredentialsError(
            f"Passwords are {MIN_PASSWORD_LENGTH} to {MAX_PASSWORD_LENGTH} characters. "
            "A few unrelated words make a long password that's easy to remember."
        )


def has_users() -> bool:
    with engine.connect() as conn:
        return bool(conn.execute(select(func.count()).select_from(users_table)).scalar())


def create_user(username: str, password: str, role: Role, *, now: float) -> User:
    username = normalize_username(username)
    check_new_credentials(username, password)
    with engine.begin() as conn:
        result = conn.execute(
            insert(users_table).values(
                username=username, password_hash=_hasher.hash(password), role=role, created_at=now
            )
        )
        key = result.inserted_primary_key
    if key is None or key[0] is None:
        raise RuntimeError("the database did not return an id for the new user")
    return User(id=int(key[0]), username=username, role=role)


def create_first_owner(username: str, password: str, *, now: float) -> User:
    """The first-run setup. The lock makes "no users yet" and the insert one
    step, so two browsers racing through setup can't both become owner."""
    with _setup_lock:
        if has_users():
            raise SetupClosedError
        return create_user(username, password, "owner", now=now)


def lock_remaining(failures: int, last_failure: float, now: float) -> float:
    """Seconds until another attempt is allowed: none for the first
    FREE_ATTEMPTS failures, then FIRST_LOCK_S doubling per failure, capped."""
    if failures < FREE_ATTEMPTS:
        return 0.0
    # The exponent is capped before use: 2.0 ** 1025 overflows a float.
    doublings = min(failures - FREE_ATTEMPTS, 32)
    lock = min(FIRST_LOCK_S * 2.0**doublings, MAX_LOCK_S)
    return max(0.0, last_failure + lock - now)


def authenticate(username: str, password: str, *, client: str, now: float) -> User | LockedOut | None:
    """The user on success, LockedOut while backing off, None on bad credentials."""
    with _login_lock:
        return _authenticate(normalize_username(username), password, client=client, now=now)


def _authenticate(username: str, password: str, *, client: str, now: float) -> User | LockedOut | None:
    key = (login_attempts_table.c.username == username, login_attempts_table.c.client == client)
    with engine.begin() as conn:
        attempt = conn.execute(select(login_attempts_table).where(*key)).mappings().fetchone()
        failures = 0
        if attempt is not None and now - attempt["last_failure"] < FORGET_FAILURES_AFTER_S:
            failures = attempt["failures"]
            wait = lock_remaining(failures, attempt["last_failure"], now)
            if wait > 0:
                # Refused before checking the password, so guesses made while
                # locked out can't confirm anything.
                return LockedOut(retry_after_s=wait)
        row = (
            conn.execute(select(users_table).where(users_table.c.username == username)).mappings().fetchone()
        )
        try:
            _hasher.verify(row["password_hash"] if row else _DUMMY_HASH, password[:MAX_PASSWORD_LENGTH])
            ok = row is not None
        except (VerificationError, InvalidHashError):
            ok = False
        conn.execute(delete(login_attempts_table).where(*key))
        if not ok:
            conn.execute(
                insert(login_attempts_table).values(
                    username=username, client=client, failures=failures + 1, last_failure=now
                )
            )
            return None
        assert row is not None  # ok implies a row
        if _hasher.check_needs_rehash(row["password_hash"]):
            # argon2-cffi's defaults get stronger over releases; upgrade on sign-in.
            conn.execute(
                update(users_table)
                .where(users_table.c.id == row["id"])
                .values(password_hash=_hasher.hash(password))
            )
        return User(id=row["id"], username=row["username"], role=row["role"])


def start_session(user: User, *, now: float) -> tuple[str, Session]:
    """A new session for user. Returns the token for the cookie, which is never stored."""
    token = secrets.token_urlsafe(32)
    session = Session(user=user, csrf_token=secrets.token_urlsafe(32))
    with engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                token_hash=_token_hash(token),
                user_id=user.id,
                csrf_token=session.csrf_token,
                created_at=now,
                last_seen=now,
            )
        )
    return token, session


def resolve_session(token: str, *, now: float, idle_s: float, max_s: float) -> Session | None:
    """The live session for token, refreshing its idle timer; None if it doesn't exist or has expired."""
    token_hash = _token_hash(token)
    with engine.begin() as conn:
        # Expired sessions are removed whenever any session is checked, so the
        # table never grows and an expired token can't be revived later by a
        # longer timeout setting.
        conn.execute(
            delete(sessions_table).where(
                or_(sessions_table.c.last_seen < now - idle_s, sessions_table.c.created_at < now - max_s)
            )
        )
        row = (
            conn.execute(
                select(
                    sessions_table.c.csrf_token, users_table.c.id, users_table.c.username, users_table.c.role
                )
                .join(users_table, users_table.c.id == sessions_table.c.user_id)
                .where(sessions_table.c.token_hash == token_hash)
            )
            .mappings()
            .fetchone()
        )
        if row is None:
            return None
        conn.execute(
            update(sessions_table).where(sessions_table.c.token_hash == token_hash).values(last_seen=now)
        )
    return Session(
        user=User(id=row["id"], username=row["username"], role=row["role"]), csrf_token=row["csrf_token"]
    )


def end_session(token: str) -> None:
    with engine.begin() as conn:
        conn.execute(delete(sessions_table).where(sessions_table.c.token_hash == _token_hash(token)))
