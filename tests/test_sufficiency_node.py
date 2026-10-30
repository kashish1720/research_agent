"""Tests for coverage routing and query refinement."""

from __future__ import annotations

from research_agent.budget import Budget
from research_agent.config import Settings
from research_agent.deps import Deps, RunContext
from research_agent.llm import FakeLLMClient
from research_agent.nodes.sufficiency import (
    compute_coverage,
    decide,
    make_sufficiency_node,
)
from research_agent.schemas import (
    AgentState,
    EvidenceItem,
    Plan,
    Reason,
    RefinedQueries,
)
from research_agent.tools.url_safety import UrlAllowlist
from research_agent.tracing import Tracer


def _plan(count: int = 2) -> Plan:
    return Plan.model_validate(
        {
            "restated_question": "Research solar power",
            "is_answerable": True,
            "assumptions": [],
            "sub_questions": [
                {
                    "id": f"q{index}",
                    "text": f"Research solar power aspect {index}",
                    "queries": [f"solar power aspect {index} evidence"],
                    "strategy": "search_and_fetch",
                    "time_sensitive": False,
                    "rationale": "Find evidence about this aspect",
                }
                for index in range(1, count + 1)
            ],
        }
    )


def _evidence(
    sub_question_id: str,
    domain: str,
    suffix: str = "first",
) -> EvidenceItem:
    return EvidenceItem(
        id=f"E-{sub_question_id}-{suffix}",
        sub_question_id=sub_question_id,
        source_id=f"S-{domain}-{suffix}",
        domain=domain,
        passage="Solar power evidence was measured during a research study.",
        claim="A research study measured solar power evidence.",
        snippet_only=False,
    )


def _setup(
    tmp_path,
    responses: list[object] | None = None,
    *,
    settings: Settings | None = None,
    evidence: list[EvidenceItem] | None = None,
    refine_loops: int = 0,
):  # type: ignore[no-untyped-def]
    active_settings = settings or Settings()
    active_settings.CACHE_DIR = str(tmp_path / "cache")
    fake = FakeLLMClient({"RefinedQueries": responses or []})
    deps = Deps(
        settings=active_settings,
        llm=fake,
        search_providers=[],
        fetch_factory=lambda _allowlist: lambda _url: None,  # type: ignore[return-value]
    )
    run = RunContext(
        run_id="sufficiency-test",
        budget=Budget(active_settings),
        tracer=Tracer("sufficiency-test", tmp_path),
        allowlist=UrlAllowlist(),
        fetch_fn=lambda _url: None,  # type: ignore[return-value]
    )
    state = AgentState(
        question="Research solar power",
        today="2026-10-30",
        plan=_plan(),
        evidence=evidence or [],
        pending_sq_ids=["q1", "q2"],
        refine_loops=refine_loops,
    )
    return make_sufficiency_node(deps, run), fake, run, state


def test_compute_coverage_counts_subquestions_and_distinct_domains() -> None:
    plan = _plan()
    evidence = [
        _evidence("q1", "www.example.com", "first"),
        _evidence("q1", "example.com", "second"),
        _evidence("q1", "research.gov", "third"),
    ]

    coverage = compute_coverage(plan, evidence)

    assert coverage.total == 2
    assert coverage.covered == 1
    assert coverage.per_subquestion[0].evidence_count == 3
    assert coverage.per_subquestion[0].independent_domains == 2
    assert coverage.per_subquestion[1].independent_domains == 0


def test_decide_uses_exact_coverage_and_refinement_boundaries() -> None:
    settings = Settings(MAX_REFINE_LOOPS=1)
    budget = Budget(settings)
    at_target = compute_coverage(
        _plan(5),
        [_evidence(f"q{index}", f"domain{index}.example") for index in range(1, 5)],
    )
    at_minimum = compute_coverage(_plan(2), [_evidence("q1", "example.com")])
    below_minimum = compute_coverage(_plan(5), [_evidence("q1", "example.com")])

    assert decide(at_target, refine_loops=0, settings=settings, budget=budget) == "synthesize"
    assert decide(
        at_minimum,
        refine_loops=1,
        settings=settings,
        budget=budget,
    ) == "synthesize"
    assert decide(
        below_minimum,
        refine_loops=1,
        settings=settings,
        budget=budget,
    ) == "insufficient"


def test_decide_refines_only_when_search_llm_and_time_remain(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    settings = Settings()
    budget = Budget(settings)
    below_minimum = compute_coverage(_plan(5), [])

    assert decide(
        below_minimum,
        refine_loops=0,
        settings=settings,
        budget=budget,
    ) == "refine"
    budget.searches = settings.MAX_SEARCHES
    assert decide(
        below_minimum,
        refine_loops=0,
        settings=settings,
        budget=budget,
    ) == "insufficient"
    budget.searches = 0
    budget.llm_calls = settings.MAX_LLM_CALLS
    assert decide(
        below_minimum,
        refine_loops=0,
        settings=settings,
        budget=budget,
    ) == "insufficient"
    budget.llm_calls = 0
    monkeypatch.setattr(budget, "time_left", lambda: 0.0)
    assert decide(
        below_minimum,
        refine_loops=0,
        settings=settings,
        budget=budget,
    ) == "insufficient"


def test_refinement_updates_only_uncovered_queries_and_loop_count(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    response = RefinedQueries(
        items=[
            {
                "sub_question_id": "q2",
                "queries": ["solar power aspect two new evidence"],
            }
        ]
    )
    node, fake, _run, state = _setup(
        tmp_path,
        [response],
        evidence=[_evidence("q1", "example.com")],
    )

    update = node(state)

    assert update["route"] == "refine"
    assert update["pending_sq_ids"] == ["q2"]
    assert update["refine_loops"] == 1
    assert update["plan"].sub_questions[0].queries == state.plan.sub_questions[0].queries
    assert update["plan"].sub_questions[1].queries == ["solar power aspect two new evidence"]
    assert len(fake.calls) == 1
    assert fake.calls[0].role == "cheap"
    assert "solar power aspect 2" in fake.calls[0].user
    assert "previous_queries" in fake.calls[0].user


def test_refinement_rejects_repeated_query_then_accepts_correction(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    old_query = _plan().sub_questions[1].queries[0]
    node, fake, _run, state = _setup(
        tmp_path,
        [
            {"items": [{"sub_question_id": "q2", "queries": [old_query]}]},
            {
                "items": [
                    {
                        "sub_question_id": "q2",
                        "queries": ["solar power aspect two distinct sources"],
                    }
                ]
            },
        ],
        evidence=[_evidence("q1", "example.com")],
    )

    update = node(state)

    assert len(fake.calls) == 2
    assert "repeats a previous query" in fake.calls[1].user
    assert update["route"] == "refine"


def test_refinement_rejects_unknown_subquestion_ids(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    node, fake, _run, state = _setup(
        tmp_path,
        [
            {"items": [{"sub_question_id": "q9", "queries": ["new solar power research"]}]},
            {"items": [{"sub_question_id": "q2", "queries": ["new solar power research"]}]},
        ],
        evidence=[_evidence("q1", "example.com")],
    )

    update = node(state)

    assert len(fake.calls) == 2
    assert "unknown sub-question id" in fake.calls[1].user
    assert update["pending_sq_ids"] == ["q2"]


def test_failed_refinement_redecides_without_refining_again(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    old_query = _plan().sub_questions[1].queries[0]
    node, fake, _run, state = _setup(
        tmp_path,
        [{"items": [{"sub_question_id": "q2", "queries": [old_query]}]}] * 3,
        evidence=[_evidence("q1", "example.com")],
    )

    update = node(state)

    assert len(fake.calls) == 3
    assert update["route"] == "synthesize"
    assert update["pending_sq_ids"] == []
    assert update["refine_loops"] == 0
    assert update["reasons"] == [Reason.UNCOVERED_SUBQUESTIONS]
    assert update["errors"] == ["Query refinement failed (ToolFailure)"]


def test_insufficient_path_makes_no_llm_call(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings()
    node, fake, run, state = _setup(
        tmp_path,
        settings=settings,
        evidence=[],
    )
    run.budget.searches = settings.MAX_SEARCHES

    update = node(state)

    assert update["route"] == "insufficient"
    assert update["reasons"] == [Reason.INSUFFICIENT_EVIDENCE]
    assert fake.calls == []


def test_refine_prompt_never_includes_page_text() -> None:
    from research_agent.prompts.refine import refine_user_prompt

    prompt = refine_user_prompt(
        [("q2", "Research regional solar power output", ["solar power output"])],
        "No previous evidence.",
    )

    assert "Research regional solar power output" in prompt
    assert "solar power output" in prompt
    assert "sentinel page text" not in prompt
