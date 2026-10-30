"""Tests for normalized Tavily and DDGS providers."""

from __future__ import annotations

import time

import pytest
from ddgs.exceptions import DDGSException, RatelimitException
from pydantic import SecretStr
from tavily.errors import BadRequestError, InvalidAPIKeyError, UsageLimitExceededError

import research_agent.tools.ddgs_search as ddgs_module
import research_agent.tools.tavily_search as tavily_module
from research_agent.config import Settings
from research_agent.errors import ToolFailure
from research_agent.schemas import ErrorClass, SearchArgs
from research_agent.tools.ddgs_search import DDGSSearchProvider
from research_agent.tools.tavily_search import TavilySearchProvider


class TavilyClientStub:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def search(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class DDGSClientStub:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[str, dict[str, object]]] = []

    def text(self, query: str, **kwargs: object) -> object:
        self.calls.append((query, kwargs))
        if self.error is not None:
            raise self.error
        return self.response


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"SEARCH_TIMEOUT_S": 2}
    values.update(overrides)
    return Settings(**values)


def test_tavily_normalizes_search_results_and_uses_basic_depth() -> None:
    client = TavilyClientStub(
        {
            "results": [
                {
                    "url": "https://example.com/a",
                    "title": "A",
                    "content": "Snippet A",
                    "score": 0.8,
                    "published_date": "2025-01-01",
                }
            ]
        }
    )
    provider = TavilySearchProvider(_settings(), client=client)

    results = provider.search(SearchArgs(query="example query"))

    assert provider.name == "tavily"
    assert results[0].provider == "tavily"
    assert results[0].snippet == "Snippet A"
    assert results[0].score == pytest.approx(0.8)
    assert results[0].published_date == "2025-01-01"
    assert client.calls[0]["search_depth"] == "basic"
    assert client.calls[0]["max_results"] == 5
    assert client.calls[0]["timeout"] == 2


def test_tavily_drops_malformed_items_and_counts_them() -> None:
    provider = TavilySearchProvider(
        _settings(),
        client=TavilyClientStub(
            {"results": [{"url": "https://example.com", "title": "missing content"}, "bad"]}
        ),
    )

    assert provider.search(SearchArgs(query="example query")) == []
    assert provider.last_dropped_count == 2


def test_tavily_missing_results_list_is_content_failure() -> None:
    provider = TavilySearchProvider(_settings(), client=TavilyClientStub({"answer": "not results"}))

    with pytest.raises(ToolFailure) as exc_info:
        provider.search(SearchArgs(query="example query"))

    assert exc_info.value.error.error_class is ErrorClass.CONTENT


def test_tavily_valid_empty_results_are_not_an_error() -> None:
    provider = TavilySearchProvider(_settings(), client=TavilyClientStub({"results": []}))

    assert provider.search(SearchArgs(query="example query")) == []
    assert provider.last_dropped_count == 0


def test_tavily_invalid_key_maps_to_client() -> None:
    provider = TavilySearchProvider(
        _settings(), client=TavilyClientStub(error=InvalidAPIKeyError("invalid"))
    )

    with pytest.raises(ToolFailure) as exc_info:
        provider.search(SearchArgs(query="example query"))

    assert exc_info.value.error.error_class is ErrorClass.CLIENT


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (UsageLimitExceededError("rate limit exceeded"), ErrorClass.TRANSIENT),
        (UsageLimitExceededError("account quota exhausted"), ErrorClass.CLIENT),
        (BadRequestError("bad request"), ErrorClass.CLIENT),
    ],
)
def test_tavily_sdk_error_classification(error: Exception, expected: ErrorClass) -> None:
    provider = TavilySearchProvider(_settings(), client=TavilyClientStub(error=error))

    with pytest.raises(ToolFailure) as exc_info:
        provider.search(SearchArgs(query="example query"))

    assert exc_info.value.error.error_class is expected


def test_tavily_construction_uses_secret_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[str | None] = []
    client = TavilyClientStub({"results": []})
    monkeypatch.setattr(
        tavily_module,
        "TavilyClient",
        lambda *, api_key: observed.append(api_key) or client,
    )
    provider = TavilySearchProvider(_settings(TAVILY_API_KEY=SecretStr("test-secret")))

    provider.search(SearchArgs(query="example query"))

    assert observed == ["test-secret"]


def test_tavily_slow_client_times_out_externally() -> None:
    class SlowClient:
        def search(self, **_kwargs: object) -> dict[str, list[object]]:
            time.sleep(0.05)
            return {"results": []}

    settings = _settings()
    settings.SEARCH_TIMEOUT_S = 0.01
    provider = TavilySearchProvider(settings, client=SlowClient())
    with pytest.raises(ToolFailure) as exc_info:
        provider.search(SearchArgs(query="example query"))

    assert exc_info.value.error.error_class is ErrorClass.TRANSIENT


def test_ddgs_normalizes_results_and_uses_max_results() -> None:
    client = DDGSClientStub(
        [
            {
                "href": "https://example.com/a",
                "title": "A",
                "body": "Snippet A",
                "date": "2025-01-01",
            }
        ]
    )
    provider = DDGSSearchProvider(_settings(), client=client)

    results = provider.search(SearchArgs(query="example query", max_results=3))

    assert provider.name == "ddgs"
    assert results[0].provider == "ddgs"
    assert results[0].snippet == "Snippet A"
    assert results[0].published_date == "2025-01-01"
    assert results[0].score is None
    assert client.calls == [("example query", {"max_results": 3})]


def test_ddgs_constructor_receives_configured_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[int] = []
    client = DDGSClientStub([])

    def make_client(*, timeout: int) -> DDGSClientStub:
        observed.append(timeout)
        return client

    monkeypatch.setattr(ddgs_module, "DDGS", make_client)
    provider = DDGSSearchProvider(_settings())
    provider.search(SearchArgs(query="example query"))

    assert observed == [2]


def test_ddgs_drops_malformed_items_and_counts_them() -> None:
    provider = DDGSSearchProvider(
        _settings(),
        client=DDGSClientStub([{"href": "https://example.com", "title": "missing body"}, None]),
    )

    assert provider.search(SearchArgs(query="example query")) == []
    assert provider.last_dropped_count == 2


def test_ddgs_missing_results_list_is_content_failure() -> None:
    provider = DDGSSearchProvider(_settings(), client=DDGSClientStub({"items": []}))

    with pytest.raises(ToolFailure) as exc_info:
        provider.search(SearchArgs(query="example query"))

    assert exc_info.value.error.error_class is ErrorClass.CONTENT


def test_ddgs_valid_empty_results_are_not_an_error() -> None:
    provider = DDGSSearchProvider(_settings(), client=DDGSClientStub([]))

    assert provider.search(SearchArgs(query="example query")) == []


def test_ddgs_rate_limit_maps_to_transient() -> None:
    provider = DDGSSearchProvider(
        _settings(), client=DDGSClientStub(error=RatelimitException("429"))
    )

    with pytest.raises(ToolFailure) as exc_info:
        provider.search(SearchArgs(query="example query"))

    assert exc_info.value.error.error_class is ErrorClass.TRANSIENT


def test_ddgs_slow_client_times_out_externally() -> None:
    class SlowClient:
        def text(self, _query: str, **_kwargs: object) -> list[object]:
            time.sleep(0.05)
            return []

    settings = _settings()
    settings.SEARCH_TIMEOUT_S = 0.01
    provider = DDGSSearchProvider(settings, client=SlowClient())
    with pytest.raises(ToolFailure) as exc_info:
        provider.search(SearchArgs(query="example query"))

    assert exc_info.value.error.error_class is ErrorClass.TRANSIENT


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("engine failed: HTTP 503", ErrorClass.TRANSIENT),
        ("engine rejected request: HTTP 400", ErrorClass.CLIENT),
        ("invalid API key", ErrorClass.CLIENT),
    ],
)
def test_ddgs_wrapped_engine_errors_are_classified(
    message: str, expected: ErrorClass
) -> None:
    provider = DDGSSearchProvider(
        _settings(), client=DDGSClientStub(error=DDGSException(message))
    )

    with pytest.raises(ToolFailure) as exc_info:
        provider.search(SearchArgs(query="example query"))

    assert exc_info.value.error.error_class is expected