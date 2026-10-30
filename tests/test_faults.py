"""Tests for deterministic fault injection wrappers."""

from __future__ import annotations

import pytest

from research_agent.errors import ToolFailure
from research_agent.schemas import ErrorClass, FetchResult, FetchStatus, SearchArgs, SearchResult
from research_agent.tools.faults import FaultInjector, wrap_fetch, wrap_search_provider


class SearchStub:
    name = "fake"
    last_dropped_count = 0

    def __init__(self) -> None:
        self.calls = 0

    def search(self, _args: SearchArgs) -> list[SearchResult]:
        self.calls += 1
        return []


def test_fault_spec_empty_and_probability_parsing() -> None:
    injector = FaultInjector("", seed=4)

    assert injector.probabilities == {}
    assert injector.counts == {"timeout": 0, "http_403": 0, "malformed": 0, "empty": 0}
    assert FaultInjector("timeout:0.3,http_403:0.2").probabilities == {
        "timeout": 0.3,
        "http_403": 0.2,
    }


@pytest.mark.parametrize("spec", ["unknown:0.2", "timeout", "timeout:not-a-number", "empty:1.2"])
def test_invalid_fault_spec_raises(spec: str) -> None:
    with pytest.raises(ValueError):
        FaultInjector(spec)


def test_seeded_faults_are_deterministic_and_counted() -> None:
    first = FaultInjector("timeout:0.5,empty:0.5", seed=9)
    second = FaultInjector("timeout:0.5,empty:0.5", seed=9)

    sequence_one = [first.next_fault() for _ in range(20)]
    sequence_two = [second.next_fault() for _ in range(20)]

    assert sequence_one == sequence_two
    assert first.counts == second.counts
    assert sum(first.counts.values()) == sum(item is not None for item in sequence_one)


@pytest.mark.parametrize(
    ("fault", "error_class"),
    [
        ("timeout", ErrorClass.TRANSIENT),
        ("http_403", ErrorClass.CLIENT),
        ("malformed", ErrorClass.CONTENT),
    ],
)
def test_search_wrapper_injects_failures(fault: str, error_class: ErrorClass) -> None:
    provider = SearchStub()
    injector = FaultInjector(f"{fault}:1", seed=1)
    wrapped = wrap_search_provider(provider, injector)

    with pytest.raises(ToolFailure) as exc_info:
        wrapped.search(SearchArgs(query="test search"))

    assert exc_info.value.error.error_class is error_class
    assert injector.counts[fault] == 1
    assert provider.calls == 0


def test_search_empty_fault_returns_empty_results() -> None:
    provider = SearchStub()
    injector = FaultInjector("empty:1", seed=1)

    assert wrap_search_provider(provider, injector).search(SearchArgs(query="test search")) == []
    assert injector.counts["empty"] == 1
    assert provider.calls == 0


@pytest.mark.parametrize("fault", ["timeout", "http_403", "malformed", "empty"])
def test_fetch_wrapper_injects_expected_outcome(fault: str) -> None:
    calls = 0

    def fetch_fn(url: str) -> FetchResult:
        nonlocal calls
        calls += 1
        return FetchResult(url=url, status=FetchStatus.OK, text="content")

    injector = FaultInjector(f"{fault}:1", seed=1)
    wrapped = wrap_fetch(fetch_fn, injector)
    if fault in {"timeout", "malformed"}:
        expected = ErrorClass.TRANSIENT if fault == "timeout" else ErrorClass.CONTENT
        with pytest.raises(ToolFailure) as exc_info:
            wrapped("https://example.com")
        assert exc_info.value.error.error_class is expected
    else:
        result = wrapped("https://example.com")
        assert result.status is (
            FetchStatus.BLOCKED if fault == "http_403" else FetchStatus.EMPTY
        )
        if fault == "http_403":
            assert result.http_status == 403
    assert injector.counts[fault] == 1
    assert calls == 0


def test_wrappers_call_through_when_no_fault_fires() -> None:
    provider = SearchStub()
    injector = FaultInjector("timeout:0", seed=1)
    assert wrap_search_provider(provider, injector).search(SearchArgs(query="test search")) == []
    assert provider.calls == 1

    result = wrap_fetch(
        lambda url: FetchResult(url=url, status=FetchStatus.OK),
        FaultInjector("", seed=1),
    )("https://example.com")
    assert result.status is FetchStatus.OK