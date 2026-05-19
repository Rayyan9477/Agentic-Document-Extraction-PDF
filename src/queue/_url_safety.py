"""V3 Phase 8 — webhook URL safety / SSRF defence.

The webhook dispatcher accepts subscriber URLs from authenticated
operators. A malicious URL can target internal services
(``http://internal-mongo:27017/``, AWS IMDSv1, GCP metadata, etc.)
unless we resolve the hostname and reject any IP that lies in a
private / loopback / link-local / multicast / metadata range.

This module exposes a single function ``check_public_url(url)`` that
parses the URL, resolves it via DNS, walks every returned address
through ``ipaddress.ip_address`` and rejects on any of:

* loopback (`127.0.0.0/8`, `::1`)
* private (`10/8`, `172.16/12`, `192.168/16`, `fc00::/7`)
* link-local (`169.254/16`, `fe80::/10`)
* multicast (`224/4`, `ff00::/8`)
* reserved (e.g. `0/8`)
* CGNAT (`100.64/10`)
* cloud metadata IPs (``169.254.169.254`` covered by link-local;
  ``fd00::ec2:0`` covered by private)

Operators who legitimately need to reach internal hosts (staging /
on-prem) can set ``WEBHOOK_ALLOW_PRIVATE=1`` to disable the rejection.
This is a deployment-wide escape hatch; per-tenant allow-lists are
deferred.

DNS-rebinding mitigation: callers should pass the resolved IP
returned by ``check_public_url`` to httpx via the ``Host`` header
trick rather than re-resolving at request time. We expose the
resolved IPs from ``check_public_url`` so the caller can do that
when it wants to.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from dataclasses import dataclass, field
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class UrlSafetyResult:
    """Outcome of ``check_public_url``."""

    allowed: bool
    reason: str | None = None
    hostname: str | None = None
    resolved_ips: tuple[str, ...] = field(default_factory=tuple)


# Cloud metadata hostnames operators sometimes try to embed when
# they don't realise the IP-level filter would catch them.
_BLOCKED_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "metadata.google.internal",
        "metadata.azure.com",
    }
)


def _is_private_or_unsafe(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> tuple[bool, str | None]:
    """Return (unsafe, reason)."""
    if addr.is_loopback:
        return True, "loopback"
    if addr.is_link_local:
        return True, "link_local"
    if addr.is_multicast:
        return True, "multicast"
    if addr.is_private:
        return True, "private"
    if addr.is_reserved:
        return True, "reserved"
    if addr.is_unspecified:
        return True, "unspecified"
    # CGNAT range 100.64.0.0/10 (RFC 6598). ``is_private`` already
    # covers this in modern Python (3.11+) but be explicit.
    try:
        cgnat = ipaddress.ip_network("100.64.0.0/10")
        if addr in cgnat:
            return True, "cgnat"
    except (ValueError, TypeError):
        pass
    return False, None


def check_public_url(url: str) -> UrlSafetyResult:
    """Validate a URL for safe outbound use; reject SSRF candidates.

    Resolves the hostname via ``socket.getaddrinfo`` and rejects when
    any returned address is private / loopback / link-local /
    multicast / reserved / CGNAT.

    Honours ``WEBHOOK_ALLOW_PRIVATE=1`` env var as an escape hatch
    for staging / on-prem deployments where webhooks legitimately
    target internal hosts.
    """
    try:
        parsed = urlparse(url)
    except Exception as e:
        return UrlSafetyResult(allowed=False, reason=f"parse_error: {e}")

    if parsed.scheme not in {"http", "https"}:
        return UrlSafetyResult(
            allowed=False,
            reason=f"unsupported_scheme: {parsed.scheme!r}",
        )

    hostname = parsed.hostname
    if not hostname:
        return UrlSafetyResult(allowed=False, reason="missing_hostname")

    # Hostname blocklist (covers the rare cases where a name maps to
    # something dangerous via /etc/hosts overrides).
    if hostname.lower() in _BLOCKED_HOSTNAMES:
        return UrlSafetyResult(
            allowed=False,
            reason=f"blocked_hostname: {hostname}",
            hostname=hostname,
        )

    # Escape hatch for staging / on-prem.
    if os.environ.get("WEBHOOK_ALLOW_PRIVATE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return UrlSafetyResult(
            allowed=True,
            reason="allow_private_override",
            hostname=hostname,
        )

    # Resolve. ``getaddrinfo`` returns both IPv4 and IPv6 results when
    # available. Any single unsafe address fails the whole URL
    # because DNS-rebinding could swap to an unsafe entry on retry.
    try:
        infos = socket.getaddrinfo(
            hostname,
            None,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as e:
        return UrlSafetyResult(
            allowed=False,
            reason=f"dns_resolution_failed: {e}",
            hostname=hostname,
        )

    resolved: list[str] = []
    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        if ip_str not in resolved:
            resolved.append(ip_str)
        try:
            addr = ipaddress.ip_address(ip_str)
        except (ValueError, TypeError):
            continue
        unsafe, reason = _is_private_or_unsafe(addr)
        if unsafe:
            return UrlSafetyResult(
                allowed=False,
                reason=f"unsafe_ip ({reason}): {ip_str}",
                hostname=hostname,
                resolved_ips=tuple(resolved),
            )

    return UrlSafetyResult(
        allowed=True,
        hostname=hostname,
        resolved_ips=tuple(resolved),
    )
