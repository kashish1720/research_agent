"""Tests for URL normalization and SSRF protections."""

from __future__ import annotations

import ipaddress
import socket

import httpx
import pytest
import respx

from research_agent.config import Settings
from research_agent.errors import ToolFailure
from research_agent.schemas import ErrorClass, FetchStatus
from research_agent.tools.fetch_page import fetch_page
from research_agent.tools.url_safety import (
    UrlAllowlist,
    _unwrap_address,
    normalize_for_allowlist,
    validate_url,
)


def _resolver_for(addresses: list[str]):
    def resolver(host: str, port: int, **_kwargs: object):
        return [
            (
                socket.AF_INET6 if ":" in address else socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                (address, port),
            )
            for address in addresses
        ]

    return resolver


def test_normalization_and_allowlist_equivalence() -> None:
    allowlist = UrlAllowlist()
    allowlist.add(["HTTPS://Example.COM:443/article#section"])

    assert normalize_for_allowlist("https://example.com/article#other") == (
        "https://example.com/article"
    )
    assert allowlist.contains("https://example.com/article")
    assert not allowlist.contains("https://example.com/other")


def test_validate_public_hostname_and_ip_literal() -> None:
    validate_url("https://example.com/path", resolver=_resolver_for(["93.184.216.34"]))
    validate_url("https://8.8.8.8/path")


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "https:///missing-host",
        "https://user:pass@example.com/",
        "http://example.com:8080/",
    ],
)
def test_unsafe_url_components_are_rejected(url: str) -> None:
    with pytest.raises(ToolFailure) as exc_info:
        validate_url(url)
    assert exc_info.value.error.error_class is ErrorClass.SECURITY


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.5",
        "169.254.169.254",
        "::1",
        "::ffff:127.0.0.1",
    ],
)
def test_non_global_ip_literals_are_rejected(address: str) -> None:
    with pytest.raises(ToolFailure) as exc_info:
        validate_url(f"http://[{address}]/" if ":" in address else f"http://{address}/")
    assert exc_info.value.error.error_class is ErrorClass.SECURITY


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.5",
        "169.254.169.254",
        "::1",
        "::ffff:127.0.0.1",
    ],
)
def test_any_non_global_dns_answer_is_rejected(address: str) -> None:
    with pytest.raises(ToolFailure) as exc_info:
        validate_url("https://example.com/", resolver=_resolver_for(["93.184.216.34", address]))
    assert exc_info.value.error.error_class is ErrorClass.SECURITY


@pytest.mark.parametrize(
    ("address", "expected_ipv4"),
    [
        ("64:ff9b::4221:3c42", "66.33.60.66"),
        ("64:ff9b::6812:19c4", "104.18.25.196"),
        ("::ffff:8.8.8.8", "8.8.8.8"),
    ],
)
def test_unwraps_nat64_and_ipv4_mapped_addresses(
    address: str,
    expected_ipv4: str,
) -> None:
    assert str(_unwrap_address(ipaddress.ip_address(address))) == expected_ipv4
    validate_url(
        f"https://[{address}]/",
        resolver=_resolver_for([address]),
    )


@pytest.mark.parametrize(
    "address",
    [
        "64:ff9b::7f00:1",
        "64:ff9b::a00:1",
        "64:ff9b::a9fe:a9fe",
        "64:ff9b::c0a8:1",
        "::ffff:127.0.0.1",
        "::1",
    ],
)
def test_private_nat64_and_mapped_addresses_are_rejected(address: str) -> None:
    with pytest.raises(ToolFailure) as exc_info:
        validate_url(f"https://[{address}]/", resolver=_resolver_for([address]))

    assert exc_info.value.error.error_class is ErrorClass.SECURITY


def test_host_with_public_ipv4_and_nat64_answers_is_allowed() -> None:
    validate_url(
        "https://mixed.example/",
        resolver=_resolver_for(["66.33.60.35", "64:ff9b::4221:3c42"]),
    )


def test_host_with_public_and_private_nat64_answers_is_blocked() -> None:
    with pytest.raises(ToolFailure) as exc_info:
        validate_url(
            "https://mixed.example/",
            resolver=_resolver_for(["66.33.60.35", "64:ff9b::7f00:1"]),
        )

    assert exc_info.value.error.error_class is ErrorClass.SECURITY


def test_redirect_to_private_nat64_host_is_blocked_before_following() -> None:
    start_url = "https://public.example/article"
    redirect_url = "https://private.example/"
    allowlist = UrlAllowlist()
    allowlist.add([start_url])
    resolved_hosts: list[str] = []

    def resolver(host: str, port: int, **_kwargs: object):
        resolved_hosts.append(host)
        address = "93.184.216.34" if host == "public.example" else "64:ff9b::7f00:1"
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (address, port))]

    with respx.mock(assert_all_called=False) as router:
        start = router.get(start_url).mock(
            return_value=httpx.Response(302, headers={"location": redirect_url})
        )
        blocked_hop = router.get(redirect_url)
        result = fetch_page(
            start_url,
            allowlist=allowlist,
            settings=Settings(),
            resolver=resolver,
        )

    assert result.status is FetchStatus.SECURITY_BLOCKED
    assert start.call_count == 1
    assert blocked_hop.call_count == 0
    assert resolved_hosts == ["public.example", "private.example"]


def test_dns_failure_is_transient() -> None:
    def resolver(_host: str, _port: int, **_kwargs: object):
        raise socket.gaierror("not found")

    with pytest.raises(ToolFailure) as exc_info:
        validate_url("https://example.com/", resolver=resolver)
    assert exc_info.value.error.error_class is ErrorClass.TRANSIENT