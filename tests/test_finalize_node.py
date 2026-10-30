"""Tests for status, confidence, and final-answer persistence."""

from __future__ import annotations

import pytest

from research_agent.budget import Budget
from research_agent.config import Settings
from research_agent.deps import Deps, RunContext
from research_agent.llm import FakeLLMClient
from research_agent.nodes.finalize import (
    compute_confidence,
    compute_status,
    make_finalize_node,
)
from research_agent.schemas import (
    AgentState,
    Confidence,
    Coverage,
    EvidenceItem,
    Finding,
    Plan,
    Reason,
    RunStatus,
    Source,
    SubQuestionCoverage,
)
from research_agent.tools.url_safety import UrlAllowlist
from research_agent.tracing import Tracer


def _coverage(
    ratio: float = 1.0,
    *,
    domains: int = 2,
) -> Coverage:
    total = 10
    covered = int(ratio * total)
    rows = [
        SubQuestionCoverage(
            id=f"q{index}",
            text=f"Topic {index}",
            evidence_count=1 if index <= covered else 0,
            independent_domains=domains if index <= covered else 0,
        )
        for index in range(1, total + 1)
    ]
    return Coverage(total=total, covered=covered, per_subquestion=rows)


def _source(
    source_id: str = "S1",
    *,
    quality: float = 0.8,
    snippet_only: bool = False,
    flags: list[str] | None = None,
) -> Source:
    return Source(
        id=source_id,
        url=f"https://{source_id.casefold()}.example/article",
        domain=f"{source_id.casefold()}.example",
        title="Research source",
        quality_score=quality,
        fetched=not snippet_only,
        snippet_only=snippet_only,
        flags=flags or [],
    )


def _finding(*, support: str = "supported") -> Finding:
    return Finding(
        statement="Research supports the stated outcome.",
        evidence_ids=["E1"],
        support=support,  # type: ignore[arg-type]
    )


def _confidence(
    *,
    coverage: Coverage | None = None,
    findings: list[Finding] | None = None,
    sources: list[Source] | None = None,
    conflicts: int = 0,
    verified: bool = True,
    status: RunStatus = RunStatus.DONE,
    settings: Settings | None = None,
):
    return compute_confidence(
        coverage=coverage or _coverage(),
        findings=[_finding()] if findings is None else findings,
        sources=[_source()] if sources is None else sources,
        conflicts_count=conflicts,
        verification_complete=verified,
        status=status,
        settings=settings or Settings(),
    )


@pytest.mark.parametrize(
    ("reasons", "answerable", "has_findings", "expected"),
    [
        ([], True, True, RunStatus.DONE),
        ([Reason.PROVIDER_FALLBACK], True, True, RunStatus.DEGRADED),
        ([Reason.INSUFFICIENT_EVIDENCE], True, False, RunStatus.FAILED_GRACEFULLY),
        ([Reason.ALL_FINDINGS_UNSUPPORTED], True, False, RunStatus.FAILED_GRACEFULLY),
        ([], False, False, RunStatus.FAILED_GRACEFULLY),
        ([Reason.NODE_ERROR], True, False, RunStatus.FAILED_GRACEFULLY),
    ],
)
def test_compute_status(
    reasons: list[Reason],
    answerable: bool,
    has_findings: bool,
    expected: RunStatus,
) -> None:
    assert (
        compute_status(reasons, answerable=answerable, has_findings=has_findings)
        is expected
    )


def test_high_confidence_requires_all_conditions_and_exact_boundaries() -> None:
    assert _confidence().level is Confidence.HIGH
    settings = Settings(HIGH_MIN_AVG_QUALITY=0.8, HIGH_MAX_SNIPPET_RATIO=0.25)
    at_boundaries = _confidence(
        sources=[
            _source("S1", quality=0.8),
            _source("S2", quality=0.8),
            _source("S3", quality=0.8),
            _source("S4", quality=0.8, snippet_only=True),
        ],
        settings=settings,
    )
    assert at_boundaries.level is Confidence.HIGH


@pytest.mark.parametrize(
    "changes",
    [
        {"coverage": _coverage(0.7)},
        {"coverage": _coverage(domains=1)},
        {"sources": [_source(quality=0.6)]},
        {"conflicts": 1},
        {
            "sources": [
                _source("S1", snippet_only=True),
                _source("S2"),
            ]
        },
        {"verified": False},
        {"status": RunStatus.DEGRADED},
    ],
)
def test_each_failed_high_condition_downgrades_to_medium(changes: dict[str, object]) -> None:
    report = _confidence(**changes)  # type: ignore[arg-type]
    assert report.level is Confidence.MEDIUM
    assert report.reasons


@pytest.mark.parametrize(
    "changes",
    [
        {"coverage": _coverage(0.6)},
        {"sources": [_source(quality=0.49)]},
        {"conflicts": 2},
    ],
)
def test_each_failed_medium_condition_downgrades_to_low(changes: dict[str, object]) -> None:
    assert _confidence(**changes).level is Confidence.LOW  # type: ignore[arg-type]


def test_confidence_low_when_empty_or_failed_and_medium_boundary_is_inclusive() -> None:
    assert _confidence(findings=[]).level is Confidence.LOW
    assert (
        _confidence(status=RunStatus.FAILED_GRACEFULLY, verified=False).level
        is Confidence.LOW
    )
    settings = Settings(MEDIUM_MIN_COVERAGE=0.7, MEDIUM_MIN_AVG_QUALITY=0.5)
    boundary = _confidence(
        coverage=_coverage(0.7),
        sources=[_source(quality=0.5)],
        verified=False,
        status=RunStatus.DEGRADED,
        settings=settings,
    )
    assert boundary.level is Confidence.MEDIUM


def _plan(*, answerable: bool = True) -> Plan:
    if not answerable:
        return Plan.model_validate(
            {
                "restated_question": "Clarify the topic",
                "is_answerable": False,
                "clarification_note": "Please specify the intended region.",
                "assumptions": [],
                "sub_questions": [],
            }
        )
    return Plan.model_validate(
        {
            "restated_question": "Research the topic",
            "is_answerable": True,
            "assumptions": ["Assume the recent full calendar year."],
            "sub_questions": [
                {
                    "id": f"q{index}",
                    "text": f"Topic aspect {index}",
                    "queries": [f"topic aspect {index} evidence"],
                    "strategy": "search_only",
                    "time_sensitive": False,
                    "rationale": "Gather evidence",
                }
                for index in (1, 2)
            ],
        }
    )


def _finalize_setup(tmp_path, state: AgentState):
    settings = Settings(
        CACHE_DIR=str(tmp_path / "cache"),
        RUNS_DIR=str(tmp_path / "runs"),
        LOG_DIR=str(tmp_path / "logs"),
    )
    deps = Deps(
        settings,
        FakeLLMClient(),
        [],
        lambda _allowlist: lambda _url: None,  # type: ignore[return-value]
    )
    run = RunContext(
        "finalize-test",
        Budget(settings),
        Tracer("finalize-test", tmp_path / "logs"),
        UrlAllowlist(),
        lambda _url: None,  # type: ignore[return-value]
    )
    return make_finalize_node(deps, run), run, deps


def _state() -> AgentState:
    return AgentState(
        question="Research the topic",
        today="2026-10-30",
        plan=_plan(),
        sources={
            "S1": _source("S1"),
            "S2": _source("S2", snippet_only=True, flags=["possible_injection"]),
        },
        evidence=[
            EvidenceItem(
                id="E1",
                sub_question_id="q1",
                source_id="S1",
                domain="s1.example",
                passage="Research outcome one is supported.",
                claim="Outcome one.",
                snippet_only=False,
            ),
            EvidenceItem(
                id="E2",
                sub_question_id="q1",
                source_id="S2",
                domain="s2.example",
                passage="Research outcome two is supported.",
                claim="Outcome two.",
                snippet_only=True,
            ),
        ],
        findings=[
            Finding(
                statement="Research outcome two is supported.",
                evidence_ids=["E2"],
                support="partial",
            ),
            Finding(
                statement="Research outcome one is supported.",
                evidence_ids=["E1"],
                support="supported",
            ),
        ],
        short_answer="Research outcomes are supported.",
        stats={"verification_complete": 1},
    )


def test_finalize_saves_valid_answer_and_only_cited_sources_in_citation_order(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    state = _state()
    node, run, deps = _finalize_setup(tmp_path, state)

    update = node(state)
    answer = update["final"]
    path = tmp_path / "runs" / f"{run.run_id}.json"
    saved = type(answer).model_validate_json(path.read_text(encoding="utf-8"))

    assert saved == answer
    assert answer.status is RunStatus.DONE
    assert [source.id for source in answer.sources_used] == ["S2", "S1"]
    assert all(source.text is None for source in answer.sources_used)
    assert any(
        "No verified evidence" in value
        for value in answer.limitations_and_assumptions
    )
    assert any("snippet" in value for value in answer.limitations_and_assumptions)
    assert any("partially supported" in value for value in answer.limitations_and_assumptions)
    assert any("Assume the recent" in value for value in answer.limitations_and_assumptions)
    assert answer.usage.est_cost_usd is None
    assert "trace_summary" in run.tracer.events[-1]["details"]
    assert deps.settings.RUNS_DIR == str(tmp_path / "runs")


def test_insufficient_and_not_answerable_answers_do_not_invent_findings(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    insufficient = AgentState(
        question="Research the topic",
        today="2026-10-30",
        plan=_plan(),
        reasons=[Reason.INSUFFICIENT_EVIDENCE],
    )
    node, _run, _deps = _finalize_setup(tmp_path, insufficient)

    answer = node(insufficient)["final"]

    assert answer.status is RunStatus.FAILED_GRACEFULLY
    assert answer.key_findings == []
    assert "Reliable information was not found" in answer.short_answer
    assert answer.suggested_next_steps
    assert any("No verified evidence" in value for value in answer.limitations_and_assumptions)


def test_failed_insufficient_answer_always_has_a_cause_limitation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    state = AgentState(
        question="Research an unspecified subject",
        today="2026-10-30",
        reasons=[Reason.INSUFFICIENT_EVIDENCE],
        abort=True,
    )
    node, _run, _deps = _finalize_setup(tmp_path, state)

    answer = node(state)["final"]

    assert answer.status is RunStatus.FAILED_GRACEFULLY
    assert answer.limitations_and_assumptions
    assert any("insufficient" in item.casefold() for item in answer.limitations_and_assumptions)


def test_not_answerable_uses_clarification_note(tmp_path) -> None:  # type: ignore[no-untyped-def]
    state = AgentState(
        question="Unclear question",
        today="2026-10-30",
        plan=_plan(answerable=False),
        reasons=[Reason.NOT_ANSWERABLE],
    )
    node, _run, _deps = _finalize_setup(tmp_path, state)

    answer = node(state)["final"]

    assert answer.key_findings == []
    assert answer.short_answer == "Please specify the intended region."
    assert answer.status is RunStatus.FAILED_GRACEFULLY
    assert answer.suggested_next_steps


def test_finalizer_internal_error_returns_and_saves_valid_fallback(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    state = _state()
    node, run, _deps = _finalize_setup(tmp_path, state)

    def fail(*_args: object, **_kwargs: object) -> Coverage:
        raise RuntimeError("internal failure")

    monkeypatch.setattr("research_agent.nodes.finalize.compute_coverage", fail)
    answer = node(state)["final"]
    saved = type(answer).model_validate_json(
        (tmp_path / "runs" / f"{run.run_id}.json").read_text(encoding="utf-8")
    )

    assert saved == answer
    assert answer.status is RunStatus.FAILED_GRACEFULLY
    assert answer.reasons[-1] is Reason.NODE_ERROR
    assert answer.key_findings == []
    assert "internal error" in answer.short_answer
