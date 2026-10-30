"""Focused tests for deterministic and batched verification."""

from __future__ import annotations

import pytest

from research_agent.budget import Budget
from research_agent.config import Settings
from research_agent.deps import Deps, RunContext
from research_agent.errors import ToolFailure
from research_agent.llm import FakeLLMClient
from research_agent.nodes.verify import make_verify_node, unmatched_numeric_tokens
from research_agent.schemas import (
    AgentState,
    Conflict,
    EntailmentOutput,
    ErrorClass,
    EvidenceItem,
    Finding,
    Plan,
    Reason,
    Source,
    ToolError,
)
from research_agent.tools.url_safety import UrlAllowlist
from research_agent.tracing import Tracer


def _plan() -> Plan:
    return Plan.model_validate(
        {
            "restated_question": "Research the topic",
            "is_answerable": True,
            "assumptions": [],
            "sub_questions": [
                {
                    "id": f"q{index}",
                    "text": f"Topic aspect {index}",
                    "queries": [f"topic aspect {index} evidence"],
                    "strategy": "search_only",
                    "time_sensitive": False,
                    "rationale": "Find evidence",
                }
                for index in (1, 2)
            ],
        }
    )


def _evidence(
    evidence_id: str = "E1",
    passage: str = "The research measured 1,000 units in 2024.",
) -> EvidenceItem:
    return EvidenceItem(
        id=evidence_id,
        sub_question_id="q1",
        source_id="S1",
        domain="example.com",
        passage=passage,
        claim="The research reports a measurement.",
        snippet_only=False,
    )


def _finding(
    statement: str = "The research measured 1000 units in 2024.",
    evidence_ids: list[str] | None = None,
) -> Finding:
    return Finding(
        statement=statement,
        evidence_ids=evidence_ids or ["E1"],
        support="unchecked",
    )


def _setup(
    tmp_path,
    responses: list[object],
    *,
    findings: list[Finding] | None = None,
    evidence: list[EvidenceItem] | None = None,
    conflicts: list[Conflict] | None = None,
    short_answer: str = "The research measured 1000 units in 2024.",
    strict_numeric: bool = True,
):  # type: ignore[no-untyped-def]
    settings = Settings(
        CACHE_DIR=str(tmp_path / "cache"),
        STRICT_NUMERIC_CHECK=strict_numeric,
    )
    fake = FakeLLMClient({"EntailmentOutput": responses})
    deps = Deps(
        settings=settings,
        llm=fake,
        search_providers=[],
        fetch_factory=lambda _allowlist: lambda _url: None,  # type: ignore[return-value]
    )
    run = RunContext(
        "verify-test",
        Budget(settings),
        Tracer("verify-test", tmp_path / "logs"),
        UrlAllowlist(),
        lambda _url: None,  # type: ignore[return-value]
    )
    state = AgentState(
        question="Research the topic",
        today="2026-10-30",
        plan=_plan(),
        sources={
            "S1": Source(
                id="S1",
                url="https://example.com",
                domain="example.com",
                title="Research",
                quality_score=0.8,
                fetched=True,
                snippet_only=False,
                text="PAGE_TEXT_SENTINEL",
            )
        },
        evidence=evidence or [_evidence()],
        findings=findings or [_finding()],
        conflicts=conflicts or [],
        short_answer=short_answer,
    )
    return make_verify_node(deps, run), fake, run, state


@pytest.mark.parametrize(
    ("statement", "passages", "unmatched"),
    [
        ("There were 1,000 units.", ["There were 1000 units."], []),
        ("The rate was 45%.", ["The rate was 45 percent."], []),
        ("The study ran in 2025.", ["The study ran in 2024."], ["2025"]),
        ("There were 75 units.", ["There were 74 units."], ["75"]),
    ],
)
def test_unmatched_numeric_tokens(
    statement: str,
    passages: list[str],
    unmatched: list[str],
) -> None:
    assert unmatched_numeric_tokens(statement, passages) == unmatched


def test_supported_finding_uses_one_batched_call_with_cited_passages_only(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    response = EntailmentOutput(
        items=[{"index": 0, "verdict": "supported", "reason": "Directly stated."}]
    )
    node, fake, _run, state = _setup(
        tmp_path,
        [response],
    )

    update = node(state)

    assert len(fake.calls) == 1
    assert fake.calls[0].role == "cheap"
    assert "PAGE_TEXT_SENTINEL" not in fake.calls[0].user
    assert "The research measured 1,000 units in 2024." in fake.calls[0].user
    assert update["findings"][0].support == "supported"
    assert update["stats"]["verification_complete"] == 1


def test_unsupported_is_dropped_and_partial_is_kept_with_limitation(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    node, _fake, _run, state = _setup(
        tmp_path,
        [
            {
                "items": [
                    {"index": 0, "verdict": "unsupported", "reason": "Not supported."},
                    {"index": 1, "verdict": "partial", "reason": "Only part is supported."},
                ]
            }
        ],
        findings=[_finding(), _finding("The research recorded 1000 units in 2024.")],
        evidence=[_evidence(), _evidence("E2")],
        short_answer="1000 units were recorded.",
    )

    update = node(state)

    assert [finding.support for finding in update["findings"]] == ["partial"]
    assert any("partially supported" in item for item in update["limitations"])
    assert Reason.FINDINGS_DROPPED in update["reasons"]


def test_duplicate_entailment_indices_trigger_correction(tmp_path) -> None:  # type: ignore[no-untyped-def]
    invalid = {
        "items": [
            {"index": 0, "verdict": "supported", "reason": "First."},
            {"index": 0, "verdict": "supported", "reason": "Duplicate."},
        ]
    }
    corrected = {
        "items": [{"index": 0, "verdict": "supported", "reason": "Valid."}]
    }
    node, fake, _run, state = _setup(tmp_path, [invalid, corrected])

    update = node(state)

    assert len(fake.calls) == 2
    assert "exactly once" in fake.calls[1].user
    assert update["findings"][0].support == "supported"


def test_numeric_mismatch_drops_finding_before_entailment(tmp_path) -> None:  # type: ignore[no-untyped-def]
    node, fake, _run, state = _setup(
        tmp_path,
        [],
        findings=[_finding("The research measured 1000 units in 2025.")],
    )

    update = node(state)

    assert fake.calls == []
    assert update["findings"] == []
    assert Reason.FINDINGS_DROPPED in update["reasons"]
    assert Reason.ALL_FINDINGS_UNSUPPORTED in update["reasons"]
    assert update["abort"] is True


def test_verification_failure_keeps_unchecked_and_runs_numeric_and_short_answer_checks(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    node, fake, _run, state = _setup(
        tmp_path,
        [
            ToolFailure(
                # The fake failure is non-retriable; deterministic validation still runs.
                error=ToolError(
                    tool="llm",
                    error_class=ErrorClass.CLIENT,
                    message="unavailable",
                )
            )
        ],
        findings=[_finding("The research measured 1000 units in 2024.")],
        short_answer="The research measured 999 units in 2025.",
    )

    update = node(state)

    assert len(fake.calls) == 1
    assert update["findings"][0].support == "unchecked"
    assert Reason.VERIFICATION_UNAVAILABLE in update["reasons"]
    assert update["short_answer"].startswith("Based on the verified findings below:")
    assert any("numeric details" in item for item in update["limitations"])


def test_invalid_ids_and_conflicts_are_filtered_deterministically(tmp_path) -> None:  # type: ignore[no-untyped-def]
    conflict = Conflict(
        topic="Comparison",
        description="Sources differ.",
        evidence_ids=["E1", "E404"],
    )
    node, fake, _run, state = _setup(
        tmp_path,
        [],
        findings=[_finding(evidence_ids=["E404"])],
        conflicts=[conflict],
    )

    update = node(state)

    assert update["findings"] == []
    assert update["conflicts"] == []
    assert fake.calls == []
    assert update["abort"] is True
