"""Tests for bounded, allowlisted HTTP fetching."""

from __future__ import annotations

import httpx
import pytest
import respx

import research_agent.tools.fetch_page as fetch_module
from research_agent.config import Settings
from research_agent.errors import ToolFailure
from research_agent.schemas import ErrorClass, FetchStatus
from research_agent.tools.fetch_page import fetch_page
from research_agent.tools.url_safety import UrlAllowlist

_URL = "https://public.example/article"
_PUBLIC_IP = "93.184.216.34"


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "FETCH_TIMEOUT_S": 2,
        "MAX_FETCH_BYTES": 2_000,
        "MAX_FETCH_TEXT_CHARS": 80,
        "MIN_PAGE_TEXT_CHARS": 20,
    }
    values.update(overrides)
    return Settings(**values)


def _allowlist(url: str = _URL) -> UrlAllowlist:
    allowlist = UrlAllowlist()
    allowlist.add([url])
    return allowlist


def _resolver(_host: str, _port: int, **_kwargs: object):
    return [(2, 1, 6, "", (_PUBLIC_IP, 443))]


def _set_extraction(
    monkeypatch: pytest.MonkeyPatch,
    text: str = "Extracted page text with enough detail to pass the minimum length.",
) -> None:
    monkeypatch.setattr(
        fetch_module,
        "_extract_html",
        lambda _html, _url: (text, "Page title", "2025-01-02"),
    )


def _mock_page(route, *, status: int = 200, headers: dict[str, str] | None = None):  # type: ignore[no-untyped-def]
    return route.mock(
        return_value=httpx.Response(
            status,
            headers={"content-type": "text/html; charset=utf-8", **(headers or {})},
            content=b"<html><body>fixture document</body></html>",
        )
    )


def test_fetch_success_extracts_metadata_truncates_and_sets_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_extraction(monkeypatch, "x" * 100)
    with respx.mock(assert_all_called=True) as router:
        route = router.get(_URL)
        _mock_page(route)

        result = fetch_page(
            _URL,
            allowlist=_allowlist(),
            settings=_settings(MAX_FETCH_TEXT_CHARS=40),
            resolver=_resolver,
        )
        user_agent = route.calls.last.request.headers["user-agent"]

    assert result.status is FetchStatus.OK
    assert result.text == "x" * 40
    assert result.truncated is True
    assert result.title == "Page title"
    assert result.published_date == "2025-01-02"
    assert user_agent == _settings().USER_AGENT


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, FetchStatus.BLOCKED),
        (403, FetchStatus.BLOCKED),
        (451, FetchStatus.BLOCKED),
        (404, FetchStatus.NOT_FOUND),
        (410, FetchStatus.NOT_FOUND),
        (400, FetchStatus.HTTP_ERROR),
    ],
)
def test_http_terminal_status_mapping(status: int, expected: FetchStatus) -> None:
    with respx.mock(assert_all_called=True) as router:
        _mock_page(router.get(_URL), status=status)
        result = fetch_page(_URL, allowlist=_allowlist(), settings=_settings(), resolver=_resolver)

    assert result.status is expected
    assert result.http_status == status


@pytest.mark.parametrize("status", [429, 500, 503])
def test_rate_limit_and_server_errors_are_transient(status: int) -> None:
    with respx.mock(assert_all_called=True) as router:
        _mock_page(router.get(_URL), status=status)
        with pytest.raises(ToolFailure) as exc_info:
            fetch_page(_URL, allowlist=_allowlist(), settings=_settings(), resolver=_resolver)

    assert exc_info.value.error.error_class is ErrorClass.TRANSIENT
    assert exc_info.value.error.http_status == status


def test_http_timeout_is_transient() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(_URL).mock(side_effect=httpx.ReadTimeout("timed out"))
        with pytest.raises(ToolFailure) as exc_info:
            fetch_page(_URL, allowlist=_allowlist(), settings=_settings(), resolver=_resolver)

    assert exc_info.value.error.error_class is ErrorClass.TRANSIENT


def test_non_html_content_is_not_extracted() -> None:
    with respx.mock(assert_all_called=True) as router:
        _mock_page(router.get(_URL), headers={"content-type": "application/pdf"})
        result = fetch_page(_URL, allowlist=_allowlist(), settings=_settings(), resolver=_resolver)

    assert result.status is FetchStatus.NON_HTML
    assert result.text == ""


def test_oversize_response_is_stopped() -> None:
    with respx.mock(assert_all_called=True) as router:
        _mock_page(
            router.get(_URL),
            headers={"content-length": "3000"},
        )
        result = fetch_page(
            _URL,
            allowlist=_allowlist(),
            settings=_settings(MAX_FETCH_BYTES=100),
            resolver=_resolver,
        )

    assert result.status is FetchStatus.TOO_LARGE


def test_near_empty_extraction_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_extraction(monkeypatch, "short")
    with respx.mock(assert_all_called=True) as router:
        _mock_page(router.get(_URL))
        result = fetch_page(_URL, allowlist=_allowlist(), settings=_settings(), resolver=_resolver)

    assert result.status is FetchStatus.EMPTY
    assert result.text == "short"


def test_url_not_allowlisted_never_makes_network_request() -> None:
    with respx.mock(assert_all_called=False) as router:
        route = router.get(_URL)
        result = fetch_page(
            _URL,
            allowlist=UrlAllowlist(),
            settings=_settings(),
            resolver=_resolver,
        )
        assert route.call_count == 0

    assert result.status is FetchStatus.SECURITY_BLOCKED


@pytest.mark.parametrize("url", ["https://user:pass@public.example/article", "http://public.example:8080/"])
def test_unsafe_allowlisted_url_is_blocked_without_network(url: str) -> None:
    with respx.mock(assert_all_called=True) as router:
        result = fetch_page(
            url,
            allowlist=_allowlist(url),
            settings=_settings(),
            resolver=_resolver,
        )

    assert result.status is FetchStatus.SECURITY_BLOCKED
    assert router.calls.call_count == 0


def test_resolved_private_address_is_blocked_without_network() -> None:
    with respx.mock(assert_all_called=True) as router:
        result = fetch_page(
            _URL,
            allowlist=_allowlist(),
            settings=_settings(),
            resolver=lambda *_args, **_kwargs: [(2, 1, 6, "", ("10.0.0.5", 443))],
        )

    assert result.status is FetchStatus.SECURITY_BLOCKED
    assert router.calls.call_count == 0


def test_redirect_to_private_ip_is_blocked_before_following() -> None:
    with respx.mock(assert_all_called=False) as router:
        start_route = router.get(_URL).mock(
            return_value=httpx.Response(302, headers={"location": "http://127.0.0.1/"})
        )
        private_route = router.get("http://127.0.0.1/")
        result = fetch_page(_URL, allowlist=_allowlist(), settings=_settings(), resolver=_resolver)
        assert start_route.call_count == 1
        assert private_route.call_count == 0

    assert result.status is FetchStatus.SECURITY_BLOCKED


def test_safe_redirect_is_followed_after_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    redirected = "https://safe.example/final"
    _set_extraction(monkeypatch)
    with respx.mock(assert_all_called=True) as router:
        first_hop = router.get(_URL).mock(
            return_value=httpx.Response(302, headers={"location": redirected})
        )
        final_hop = _mock_page(router.get(redirected))
        result = fetch_page(_URL, allowlist=_allowlist(), settings=_settings(), resolver=_resolver)
        assert first_hop.call_count == 1
        assert final_hop.call_count == 1

    assert result.status is FetchStatus.OK
    assert result.final_url == redirected


def test_redirect_limit_returns_terminal_http_error() -> None:
    redirected = "https://safe.example/next"
    with respx.mock(assert_all_called=True) as router:
        first_hop = router.get(_URL).mock(
            return_value=httpx.Response(302, headers={"location": redirected})
        )
        second_hop = router.get(redirected).mock(
            return_value=httpx.Response(302, headers={"location": _URL})
        )
        result = fetch_page(
            _URL,
            allowlist=_allowlist(),
            settings=_settings(MAX_REDIRECTS=1),
            resolver=_resolver,
        )
        assert first_hop.call_count == 1
        assert second_hop.call_count == 1

    assert result.status is FetchStatus.HTTP_ERROR
    assert result.http_status == 302


def test_extraction_exception_is_not_returned_as_page_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def extraction_failure(_html: str, _url: str) -> tuple[str, str | None, str | None]:
        raise ValueError("bad")

    monkeypatch.setattr(fetch_module, "_extract_html", extraction_failure)
    with respx.mock(assert_all_called=True) as router:
        _mock_page(router.get(_URL))
        result = fetch_page(_URL, allowlist=_allowlist(), settings=_settings(), resolver=_resolver)

    assert result.status is FetchStatus.EMPTY
    assert result.text == ""
    assert result.error is not None
    assert "bad" not in result.text