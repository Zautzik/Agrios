"""Who may connect, and cross-site request forgery (CSRF) protection.

CSRF follows the OWASP CSRF Prevention Cheat Sheet, defence in depth:
- a synchronizer token on every state-changing request: the session's own
  token once signed in, a double-submit cookie on the sign-in and setup forms;
- a Fetch Metadata check (Sec-Fetch-Site) and an Origin check, which reject
  cross-site requests before the token is looked at;
- SameSite=Strict cookies.
"""

from __future__ import annotations

import ipaddress
import secrets
from collections.abc import Iterable

from fastapi import Request

from agrios.services.settings import AccessSettings

IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network
IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

# "lan" means these ranges, listed explicitly rather than ipaddress's
# is_private, whose meaning has changed between Python releases.
LAN_NETWORKS: tuple[IPNetwork, ...] = tuple(
    ipaddress.ip_network(n)
    for n in (
        "10.0.0.0/8",  # RFC 1918
        "172.16.0.0/12",  # RFC 1918
        "192.168.0.0/16",  # RFC 1918
        "169.254.0.0/16",  # RFC 3927 link-local
        "fc00::/7",  # RFC 4193 unique local
        "fe80::/10",  # RFC 4291 link-local
    )
)

CSRF_COOKIE = "agrios_csrf"
CSRF_FIELD = "csrf_token"
CSRF_HEADER = "x-csrf-token"


def parse_client(host: str | None) -> IPAddress | None:
    if host is None:
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return None
    # An IPv4 client on a dual-stack socket arrives as ::ffff:a.b.c.d.
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def _in_any(address: IPAddress, networks: Iterable[IPNetwork]) -> bool:
    return any(address.version == n.version and address in n for n in networks)


def client_allowed(host: str | None, access: AccessSettings) -> bool:
    """Whether a client may reach the console at all. Fails closed: an address
    that can't be parsed is refused. This machine is always allowed, so a
    wrong allow list can always be fixed from the console machine itself."""
    address = parse_client(host)
    if address is None:
        return False
    if address.is_loopback:
        return True
    if access.exposure == "lan":
        return _in_any(address, LAN_NETWORKS)
    if access.exposure == "custom":
        return _in_any(address, access.allowed_networks)
    return False


def host_allowed(host_header: str | None, access: AccessSettings) -> bool:
    """Whether the Host header names this console. A DNS-rebinding attack makes
    an attacker's domain resolve to this machine, so the browser connects from
    loopback and sends matching Host and Origin headers, passing the address and
    CSRF checks; the one thing it can't do is name an address or a host the
    owner listed. So: IP literals, localhost, and the configured names only."""
    if not host_header:
        return False
    host = host_header.strip().lower()
    if host.startswith("["):  # [IPv6]:port
        host = host[1 : host.find("]")] if "]" in host else ""
    elif host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    if host == "localhost" or host in access.hostnames:
        return True
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def bind_host(access: AccessSettings) -> str:
    """The address to listen on. "local" doesn't listen on the network at
    all, which is stronger than accepting connections and refusing them."""
    return "127.0.0.1" if access.exposure == "local" else "0.0.0.0"  # exposure is the owner's choice


def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def cross_site(request: Request) -> bool:
    """True if the browser says the request came from another site."""
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site is not None and fetch_site not in ("same-origin", "none"):
        return True
    origin = request.headers.get("origin")
    if origin is not None and origin != "null":
        host = request.headers.get("host", "")
        return origin.removeprefix("http://").removeprefix("https://") != host
    return origin == "null"


def csrf_valid(submitted: str | None, expected: str | None) -> bool:
    if not submitted or not expected:
        return False
    return secrets.compare_digest(submitted.encode(), expected.encode())
