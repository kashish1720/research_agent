"""Tests for evidence-only strong-model synthesis."""

from __future__ import annotations

from research_agent.budget import Budget
from research_agent.config import Settings
from research_agent.deps import Deps, RunContext
from research_agent.llm import FakeLLMClient
from research_agent.nodes.synthesize import make_synthesize_node
from research_agent.schemas import (
    AgentState,
    EvidenceItem,
    Plan,
    Reason,
    Source,
    SynthesisDraft,
)
from research_agent.tools.url_safety import UrlAllowlist
from research_agent.tracing import Tracer

_SENTINEL = "PRIVATE_PAGE_SENTINEL_NOT_EVIDENCE"
_PASSAGE = "Solar output increased steadily across the measured study period."


def _plan() -> Plan:
    return Plan.model_validate(
        {
            "restated_question": "Research solar power",
            "is_answerable": True,
            "assumptions": [],
            "sub_questions": [
                {
                    "id": "q1",
                    "text": "Research solar power production",
                    "queries": ["solar power production study"],
                    "strategy": "search_and_fetch",
                    "time_sensitive": True,
                    "rationale": "Review measured evidence",
                },
                {
                    "id": "q2",
                    "text": "Research solar power regional output",
                    "queries": ["solar power regional output research"],
                    "strategy": "search_and_fetch",
                    "time_sensitive": False,
                    "rationale": "Review regional evidence",
                },
            ],
        }
    )


def _evidence(
    evidence_id: str,
    sub_question_id: str = "q1",
    source_id: str = "S1",
    passage: str = _PASSAGE,
) -> EvidenceItem:
    return EvidenceItem(
        id=evidence_id,
        sub_question_id=sub_question_id,
        source_id=source_id,
        domain="example.com",
        published_date="2025-01-01",
        passage=passage,
        claim="The study reports an increase in solar output.",
        snippet_only=False,
    )


def _draft(evidence_ids: list[str] | None = None) -> SynthesisDraft:
    return SynthesisDraft(
        short_answer="Solar output increased in the cited study. More evidence is needed.",
        findings=[
            {
                "statement": "Solar output increased during the measured study period.",
                "evidence_ids": evidence_ids or ["E1"],
            }
        ],
        conflicts=[],
        limitations=["Evidence is limited to the available sources."],
        next_steps=["Compare results across additional regions.", "Review newer studies."],
    )


def _setup(
    tmp_path,
    responses: list[object],
    *,
    evidence: list[EvidenceItem] | None = None,
    sources: dict[str, Source] | None = None,
    settings: Settings | None = None,
):  # type: ignore[no-untyped-def]
    active_settings = settings or Settings()
    active_settings.CACHE_DIR = str(tmp_path / "cache")
    fake = FakeLLMClient({"SynthesisDraft": responses})
    deps = Deps(
        settings=active_settings,
        llm=fake,
        search_providers=[],
        fetch_factory=lambda _allowlist: lambda _url: None,  # type: ignore[return-value]
    )
    run = RunContext(
        run_id="synthesize-test",
        budget=Budget(active_settings),
        tracer=Tracer("synthesize-test", tmp_path),
        allowlist=UrlAllowlist(),
        fetch_fn=lambda _url: None,  # type: ignore[return-value]
    )
    state = AgentState(
        question="Research solar power",
        today="2026-10-30",
        plan=_plan(),
        evidence=evidence if evidence is not None else [_evidence("E1")],
        sources=sources or {},
    )
    return make_synthesize_node(deps, run), fake, run, state


def test_synthesis_success_uses_only_evidence_fields_and_maps_state(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    source = Source(
        id="S1",
        url="https://example.com/source",
        domain="example.com",
        title=f"Title includes {_SENTINEL}",
        quality_score=0.8,
        fetched=True,
        snippet_only=False,
        text=f"Full page includes {_SENTINEL}",
    )
    node, fake, _run, state = _setup(
        tmp_path,
        [_draft()],
        sources={"S1": source},
    )

    update = node(state)

    assert len(fake.calls) == 1
    assert fake.calls[0].role == "strong"
    assert fake.calls[0].schema is SynthesisDraft
    assert _SENTINEL not in fake.calls[0].user
    assert "Full page" not in fake.calls[0].user
    assert "Solar output increased steadily" in fake.calls[0].user
    assert "For time-sensitive questions, mention the dates" in fake.calls[0].system
    assert update["draft"] == _draft()
    assert update["findings"][0].support == "unchecked"
    assert update["conflicts"] == []
    assert update["limitations"] == _draft().limitations
    assert update["next_steps"] == _draft().next_steps
    assert update["short_answer"] == _draft().short_answer
    assert update["abort"] is False


def test_unknown_evidence_id_is_corrected_by_resilient_llm(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    node, fake, _run, state = _setup(
        tmp_path,
        [_draft(["E99"]), _draft(["E1"])],
    )

    update = node(state)

    assert len(fake.calls) == 2
    assert "unknown evidence ids" in fake.calls[1].user
    assert update["findings"][0].evidence_ids == ["E1"]


def test_synthesis_evidence_cap_keeps_best_per_question_deterministically(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    evidence = [
        _evidence("E1", "q1", "S1"),
        _evidence("E2", "q1", "S2", "Solar output increased steadily during the study period."),
        _evidence("E3", "q2", "S3", "Regional solar production varied across study locations."),
        _evidence("E4", "q2", "S4", "The analysis reports regional power output levels."),
    ]
    sources = {
        "S1": Source(
            id="S1",
            url="https://a.example/one",
            domain="a.example",
            title="one",
            quality_score=0.2,
            fetched=True,
            snippet_only=False,
        ),
        "S2": Source(
            id="S2",
            url="https://b.example/two",
            domain="b.example",
            title="two",
            quality_score=0.9,
            fetched=True,
            snippet_only=False,
        ),
        "S3": Source(
            id="S3",
            url="https://c.example/three",
            domain="c.example",
            title="three",
            quality_score=0.8,
            fetched=True,
            snippet_only=False,
        ),
        "S4": Source(
            id="S4",
            url="https://d.example/four",
            domain="d.example",
            title="four",
            quality_score=0.1,
            fetched=True,
            snippet_only=False,
        ),
    }
    settings = Settings(MAX_EVIDENCE_FOR_SYNTH=2)
    node, fake, _run, state = _setup(
        tmp_path,
        [_draft(["E2", "E3"])],
        evidence=evidence,
        sources=sources,
        settings=settings,
    )

    node(state)

    assert '"id": "E2"' in fake.calls[0].user
    assert '"id": "E3"' in fake.calls[0].user
    assert '"id": "E1"' not in fake.calls[0].user
    assert '"id": "E4"' not in fake.calls[0].user


def test_zero_findings_is_rejected_and_corrected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    invalid = {
        "short_answer": "No findings.",
        "findings": [],
        "conflicts": [],
        "limitations": [],
        "next_steps": [],
    }
    node, fake, _run, state = _setup(tmp_path, [invalid, _draft()])

    node(state)

    assert len(fake.calls) == 2
    assert "at least one finding" in fake.calls[1].user


def test_conflict_requires_two_valid_evidence_ids(tmp_path) -> None:  # type: ignore[no-untyped-def]
    invalid = {
        **_draft().model_dump(),
        "conflicts": [
            {
                "topic": "Regional output",
                "description": "Sources disagree.",
                "evidence_ids": ["E1", "E99"],
            }
        ],
    }
    corrected = {
        **_draft().model_dump(),
        "conflicts": [
            {
                "topic": "Regional output",
                "description": "Sources disagree.",
                "evidence_ids": ["E1", "E2"],
            }
        ],
    }
    node, fake, _run, state = _setup(
        tmp_path,
        [invalid, corrected],
        evidence=[_evidence("E1"), _evidence("E2", "q2", "S2")],
    )

    update = node(state)

    assert len(fake.calls) == 2
    assert "at least two valid evidence ids" in fake.calls[1].user
    assert update["conflicts"][0].evidence_ids == ["E1", "E2"]


def test_synthesis_failure_aborts_without_fallback_answer(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    invalid = _draft(["E99"])
    node, fake, _run, state = _setup(tmp_path, [invalid, invalid, invalid])

    update = node(state)

    assert len(fake.calls) == 3
    assert update["draft"] is None
    assert update["short_answer"] is None
    assert update["findings"] == []
    assert update["abort"] is True
    assert Reason.NODE_ERROR in update["reasons"]
    assert update["errors"] == ["Synthesis failed (ToolFailure)"]
