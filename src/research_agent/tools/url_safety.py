"""URL normalization, allowlisting, and SSRF-resistant address validation."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterable
from typing import Any, TypeAlias
from urllib.parse import SplitResult, urlsplit, urlunsplit

from research_agent.errors import ToolFailure
from research_agent.schemas import ErrorClass, ToolError

AddressInfo: TypeAlias = tuple[int, int, int, str, tuple[Any, ...]]
Resolver: TypeAlias = Any
_NAT64_IPV4_PREFIX = ipaddress.ip_network("64:ff9b::/96")


def normalize_for_allowlist(url: str) -> str:
    """Normalize URL scheme/host/default port and remove fragments."""
    try:
        parts = urlsplit(url.strip())
        hostname = parts.hostname
        if not hostname:
            return url.strip()
        host = hostname.rstrip(".").encode("idna").decode("ascii").lower()
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if isinstance(address, ipaddress.IPv6Address):
            host = f"[{host}]"
        userinfo = parts.netloc.rsplit("@", 1)[0] + "@" if "@" in parts.netloc else ""
        port = parts.port
        scheme = parts.scheme.lower()
        default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
        netloc = userinfo + host
        if port is not None and not default_port:
            netloc = f"{netloc}:{port}"
        return urlunsplit(
            SplitResult(scheme, netloc, parts.path or "/", parts.query, "")
        )
    except (UnicodeError, ValueError):
        return url.strip()


class UrlAllowlist:
    """Run-local set of normalized search-result URLs."""

    def __init__(self) -> None:
        self._urls: set[str] = set()

    def add(self, urls: Iterable[str]) -> None:
        self._urls.update(normalize_for_allowlist(url) for url in urls)

    def contains(self, url: str) -> bool:
        return normalize_for_allowlist(url) in self._urls


def _failure(error_class: ErrorClass, message: str) -> ToolFailure:
    return ToolFailure(ToolError(tool="fetch", error_class=error_class, message=message))


def _unwrap_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Unwrap IPv4-compatible representations before applying address policy."""
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return address.ipv4_mapped
        if address in _NAT64_IPV4_PREFIX:
            return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return address


def _is_non_global(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    address = _unwrap_address(address)
    return (
        not address.is_global
        or address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _check_address(address_text: str) -> None:
    try:
        address = ipaddress.ip_address(address_text.split("%", 1)[0])
    except ValueError as err:
        raise _failure(ErrorClass.SECURITY, "DNS returned an invalid IP address") from err
    if _is_non_global(address):
        raise _failure(ErrorClass.SECURITY, "URL resolves to a non-global IP address")


def validate_url(url: str, resolver: Resolver = socket.getaddrinfo) -> None:
    """Reject unsafe URLs and any hostname resolving to a non-global address."""
    try:
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        host = parts.hostname
        port = parts.port
    except ValueError as err:
        raise _failure(ErrorClass.SECURITY, "URL is malformed") from err

    if scheme not in {"http", "https"}:
        raise _failure(ErrorClass.SECURITY, "URL scheme must be http or https")
    if not host:
        raise _failure(ErrorClass.SECURITY, "URL must include a host")
    if "@" in parts.netloc:
        raise _failure(ErrorClass.SECURITY, "URL userinfo is not allowed")
    if port is not None and port not in {80, 443}:
        raise _failure(ErrorClass.SECURITY, "URL port must be 80 or 443")

    try:
        ip_address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        ip_address = None
    if ip_address is not None:
        if _is_non_global(ip_address):
            raise _failure(ErrorClass.SECURITY, "URL uses a non-global IP address")
        return

    try:
        addresses = resolver(
            host,
            port or (443 if scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except OSError as err:
        raise _failure(ErrorClass.TRANSIENT, "DNS resolution failed") from err
    if not addresses:
        raise _failure(ErrorClass.TRANSIENT, "DNS resolution returned no addresses")
    for address_info in addresses:
        try:
            address_text = str(address_info[4][0])
        except (IndexError, TypeError) as err:
            raise _failure(ErrorClass.TRANSIENT, "DNS resolution returned invalid data") from err
        _check_address(address_text)