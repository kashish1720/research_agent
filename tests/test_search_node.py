"""Tests for sequential search, argument repair, and provider fallback."""

from __future__ import annotations

from collections.abc import Callable

from research_agent.budget import Budget
from research_agent.config import Settings
from research_agent.deps import Deps, RunContext
from research_agent.errors import ToolFailure
from research_agent.llm import FakeLLMClient
from research_agent.nodes.search import make_search_node
from research_agent.schemas import (
    AgentState,
    ErrorClass,
    Plan,
    Reason,
    SearchArgs,
    SearchResult,
    ToolError,
)
from research_agent.tools.url_safety import UrlAllowlist
from research_agent.tracing import Tracer


def _plan(queries: list[str] | None = None) -> Plan:
    return Plan.model_validate(
        {
            "restated_question": "Research solar power",
            "is_answerable": True,
            "assumptions": [],
            "sub_questions": [
                {
                    "id": "q1",
                    "text": "Research solar power generation methods",
                    "queries": queries or ["solar power generation research"],
                    "strategy": "search_and_fetch",
                    "time_sensitive": False,
                    "rationale": "Find reliable evidence about solar power",
                }
            ],
        }
    )


class FakeProvider:
    def __init__(
        self,
        name: str,
        response: list[SearchResult] | Callable[[SearchArgs], list[SearchResult]],
        error: ToolFailure | None = None,
    ) -> None:
        self.name = name
        self.response = response
        self.error = error
        self.calls: list[SearchArgs] = []

    def search(self, args: SearchArgs) -> list[SearchResult]:
        self.calls.append(args)
        if self.error is not None:
            raise self.error
        if callable(self.response):
            return self.response(args)
        return self.response


def _result(url: str = "https://example.com/solar") -> SearchResult:
    return SearchResult(
        url=url,
        title="Solar power generation research",
        snippet="Research findings describe solar power generation methods.",
        score=0.8,
        provider="fake",
    )


def _setup(
    tmp_path,
    providers: list[FakeProvider],
    *,
    plan: Plan | None = None,
    fake_llm: FakeLLMClient | None = None,
    settings: Settings | None = None,
):  # type: ignore[no-untyped-def]
    active_settings = settings or Settings()
    active_settings.CACHE_DIR = str(tmp_path / "cache")
    llm = fake_llm or FakeLLMClient()
    deps = Deps(
        settings=active_settings,
        llm=llm,
        search_providers=providers,
        fetch_factory=lambda _allowlist: lambda _url: None,  # type: ignore[return-value]
    )
    run = RunContext(
        run_id="search-test",
        budget=Budget(active_settings),
        tracer=Tracer("search-test", tmp_path),
        allowlist=UrlAllowlist(),
        fetch_fn=lambda _url: None,  # type: ignore[return-value]
    )
    state = AgentState(
        question="Research solar power",
        today="2026-10-30",
        plan=plan or _plan(),
        pending_sq_ids=["q1"],
    )
    return make_search_node(deps, run), state, run, llm


def test_search_node_normalizes_args_and_annotates_results(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    provider = FakeProvider("fake", [_result()])
    node, state, run, _ = _setup(tmp_path, [provider])

    update = node(state)
    result = update["search_results"]["q1"][0]

    assert provider.calls == [SearchArgs(query="solar power generation research")]
    assert result.sub_question_id == "q1"
    assert result.query == provider.calls[0].query
    assert run.allowlist.contains(result.url)


def test_invalid_query_is_repaired_by_cheap_llm_without_double_budgeting(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    valid_plan = _plan()
    invalid_subquestion = valid_plan.sub_questions[0].model_copy(
        update={"queries": ["https://invalid.example/query"]}
    )
    invalid_plan = valid_plan.model_copy(update={"sub_questions": [invalid_subquestion]})
    provider = FakeProvider("fake", [_result()])
    fake_llm = FakeLLMClient(
        {
            "SearchArgs": [
                {
                    "query": "solar power generation evidence",
                    "max_results": 5,
                    "depth": "basic",
                }
            ]
        }
    )
    node, state, run, _ = _setup(
        tmp_path,
        [provider],
        plan=invalid_plan,
        fake_llm=fake_llm,
    )

    update = node(state)

    assert provider.calls[0].query == "solar power generation evidence"
    assert update["search_results"]["q1"][0].query == provider.calls[0].query
    assert run.budget.llm_calls == 1
    assert fake_llm.calls[0].role == "cheap"


def test_exhausted_query_repairs_fall_back_to_subquestion_text(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    valid_plan = _plan()
    invalid_subquestion = valid_plan.sub_questions[0].model_copy(
        update={"queries": ["https://invalid.example/query"]}
    )
    invalid_plan = valid_plan.model_copy(update={"sub_questions": [invalid_subquestion]})
    invalid_args = {
        "query": "https://still-invalid.example/query",
        "max_results": 5,
        "depth": "basic",
    }
    fake_llm = FakeLLMClient({"SearchArgs": [invalid_args, invalid_args]})
    provider = FakeProvider("fake", [_result()])
    settings = Settings(MAX_CORRECTIONS=1)
    node, state, _, _ = _setup(
        tmp_path,
        [provider],
        plan=invalid_plan,
        fake_llm=fake_llm,
        settings=settings,
    )

    node(state)

    assert provider.calls[0].query == "Research solar power generation methods"
    assert len(fake_llm.calls) == 2


def test_client_failure_falls_back_to_next_provider_and_adds_reason(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    tavily = FakeProvider(
        "tavily",
        [],
        error=ToolFailure(
            ToolError(tool="tavily", error_class=ErrorClass.CLIENT, message="client error")
        ),
    )
    ddgs = FakeProvider("ddgs", [_result("https://example.com/fallback")])
    node, state, run, _ = _setup(tmp_path, [tavily, ddgs])

    update = node(state)

    assert len(tavily.calls) == 1
    assert len(ddgs.calls) == 1
    assert update["reasons"] == [Reason.PROVIDER_FALLBACK]
    assert run.allowlist.contains("https://example.com/fallback")


def test_search_budget_exhaustion_keeps_partial_results(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    provider = FakeProvider(
        "fake",
        [_result()],
    )
    plan = _plan(
        [
            "solar power generation research",
            "solar power technology evidence",
        ]
    )
    settings = Settings(MAX_SEARCHES=1)
    node, state, _, _ = _setup(tmp_path, [provider], plan=plan, settings=settings)

    update = node(state)

    assert len(update["search_results"]["q1"]) == 1
    assert update["search_results"]["q1"][0].url == "https://example.com/solar"
    assert Reason.BUDGET_EXHAUSTED in update["reasons"]


def test_all_provider_failures_are_noted_and_search_continues(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    provider = FakeProvider(
        "fake",
        [],
        error=ToolFailure(
            ToolError(tool="fake", error_class=ErrorClass.CLIENT, message="rejected")
        ),
    )
    node, state, run, _ = _setup(tmp_path, [provider])

    update = node(state)

    assert update["search_results"] == {}
    assert update["errors"] == ["All search providers failed for q1"]
    assert any(
        isinstance(event["details"], dict)
        and event["details"].get("action") == "all_providers_failed"
        for event in run.tracer.events
    )
