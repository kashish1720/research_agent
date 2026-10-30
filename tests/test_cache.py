"""Tests for the successful-result disk cache."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from research_agent.budget import Budget
from research_agent.cache import CachedSearchProvider, DiskCache
from research_agent.config import Settings, load_settings
from research_agent.deps import Deps
from research_agent.errors import ToolFailure
from research_agent.llm import FakeLLMClient, ResilientLLM
from research_agent.schemas import (
    ErrorClass,
    FetchResult,
    FetchStatus,
    Plan,
    SearchArgs,
    SearchResult,
    ToolError,
)
from research_agent.tools.url_safety import UrlAllowlist
from research_agent.tracing import Tracer


def _plan() -> Plan:
    return Plan.model_validate(
        {
            "restated_question": "Research one topic",
            "is_answerable": True,
            "assumptions": [],
            "sub_questions": [
                {
                    "id": "q1",
                    "text": "Research one topic",
                    "queries": ["research one topic"],
                    "strategy": "search_only",
                    "time_sensitive": False,
                    "rationale": "Gather evidence",
                }
            ],
        }
    )


def _llm(
    settings: Settings,
    client: FakeLLMClient,
    tmp_path,
    run_id: str,
) -> tuple[ResilientLLM, Budget, Tracer]:  # type: ignore[no-untyped-def]
    budget = Budget(settings)
    tracer = Tracer(run_id, tmp_path / "logs")
    return ResilientLLM(client, settings, budget, tracer), budget, tracer


def test_llm_hit_returns_parsed_output_without_budget_or_call(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(CACHE_DIR=str(tmp_path / "cache"))
    first_client = FakeLLMClient({"Plan": [_plan()]})
    first, _first_budget, _first_tracer = _llm(settings, first_client, tmp_path, "first")
    expected = first.structured(
        role="cheap", system="system", user="question", schema=Plan
    ).parsed

    second_client = FakeLLMClient()
    second, budget, tracer = _llm(settings, second_client, tmp_path, "second")
    actual = second.structured(
        role="cheap", system="system", user="question", schema=Plan
    ).parsed

    assert actual == expected
    assert second_client.calls == []
    assert budget.llm_calls == 0
    assert budget.input_tokens == budget.output_tokens == 0
    assert any(event["details"].get("cache_hit") is True for event in tracer.events)


def test_llm_failure_is_not_cached(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(CACHE_DIR=str(tmp_path / "cache"))

    def fail(_call: object) -> object:
        raise ToolFailure(
            ToolError(
                tool="llm",
                error_class=ErrorClass.CLIENT,
                message="expected fake failure",
            )
        )

    failed, _budget, _tracer = _llm(
        settings,
        FakeLLMClient({"Plan": [fail]}),
        tmp_path,
        "failed",
    )
    with pytest.raises(ToolFailure):
        failed.structured(role="cheap", system="system", user="question", schema=Plan)
    assert not list((tmp_path / "cache").rglob("*.json"))

    succeeding_client = FakeLLMClient({"Plan": [_plan()]})
    succeeding, _budget, _tracer = _llm(
        settings, succeeding_client, tmp_path, "succeeding"
    )
    succeeding.structured(role="cheap", system="system", user="question", schema=Plan)
    assert len(succeeding_client.calls) == 1


def test_fault_mode_bypasses_cache_read_and_write(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(
        CACHE_DIR=str(tmp_path / "cache"),
        FAULT_INJECT="timeout:0",
    )
    for run_id in ("fault-one", "fault-two"):
        client = FakeLLMClient({"Plan": [_plan()]})
        llm, budget, _tracer = _llm(settings, client, tmp_path, run_id)
        llm.structured(role="cheap", system="system", user="question", schema=Plan)
        assert len(client.calls) == 1
        assert budget.llm_calls == 1
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_search_provider_miss_then_hit_and_trace(tmp_path) -> None:  # type: ignore[no-untyped-def]
    class SearchStub:
        name = "fake"

        def __init__(self) -> None:
            self.calls = 0

        def search(self, args: SearchArgs) -> list[SearchResult]:
            self.calls += 1
            return [
                SearchResult(
                    url="https://example.com/result",
                    title=args.query,
                    snippet="A result",
                    provider="fake",
                )
            ]

    underlying = SearchStub()
    provider = CachedSearchProvider(
        underlying,
        DiskCache(tmp_path / "cache"),
    )
    first = provider.search(SearchArgs(query="research topic"))
    tracer = Tracer("cache-search", tmp_path / "logs")
    budget = Budget(Settings())
    provider.set_run_context(tracer, budget)
    budget.consume_search()
    second = provider.search(SearchArgs(query="research topic"))

    assert second == first
    assert underlying.calls == 1
    assert budget.searches == 0
    assert any(event["details"].get("cache_hit") is True for event in tracer.events)


def test_fetch_cache_hit_skips_fetch_and_fault_disables_cache(tmp_path) -> None:  # type: ignore[no-untyped-def]
    calls = 0

    def fetch(_allowlist: UrlAllowlist):
        def run(url: str) -> FetchResult:
            nonlocal calls
            calls += 1
            return FetchResult(url=url, status=FetchStatus.OK, text="cached page")

        return run

    settings = Settings(CACHE_DIR=str(tmp_path / "cache"))
    deps = Deps(settings, FakeLLMClient(), [], fetch)
    first = deps.new_run("first")
    expected = first.fetch_fn("https://example.com/page")
    second = deps.new_run("second")
    second.budget.consume_fetch()
    assert second.fetch_fn("https://example.com/page") == expected
    assert calls == 1
    assert second.budget.fetches == 0
    assert any(event["details"].get("cache_hit") is True for event in second.tracer.events)

    fault_settings = settings.model_copy(update={"FAULT_INJECT": "timeout:0"})
    fault_deps = Deps(fault_settings, FakeLLMClient(), [], fetch)
    fault_deps.new_run("fault-one").fetch_fn("https://example.com/page")
    fault_deps.new_run("fault-two").fetch_fn("https://example.com/page")
    assert calls == 3


def test_no_cache_flag_and_environment_disable_cache(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from scripts.smoke_llm import _apply_cli_options, _parse_args

    args = _parse_args(["research question", "--no-cache"])
    settings = load_settings({"USE_CACHE": "false"})
    cli_settings = _apply_cli_options(
        Settings(CACHE_DIR=str(tmp_path / "cache")),
        no_cache=args.no_cache,
    )
    for run_id in ("no-cache-one", "no-cache-two"):
        client = FakeLLMClient({"Plan": [_plan()]})
        llm, _budget, _tracer = _llm(cli_settings, client, tmp_path, run_id)
        llm.structured(role="cheap", system="system", user="question", schema=Plan)
        assert len(client.calls) == 1

    assert args.no_cache is True
    assert settings.USE_CACHE is False
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_cache_key_is_canonical_and_result_models_validate(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cache = DiskCache(tmp_path)
    cache.store("custom", {"b": 2, "a": 1}, {"value": "ok"})
    found, result = cache.load("custom", {"a": 1, "b": 2})

    class CachedValue(BaseModel):
        value: str

    assert found
    assert CachedValue.model_validate(result).value == "ok"
