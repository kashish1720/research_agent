"""Tests for src/research_agent/schemas.py."""

from __future__ import annotations

import json

import pytest

from research_agent.schemas import (
    AgentState,
    Confidence,
    ConfidenceReport,
    Conflict,
    Coverage,
    EntailmentItem,
    EntailmentOutput,
    ErrorClass,
    EvidenceItem,
    ExtractionItem,
    ExtractionOutput,
    FetchResult,
    FetchStatus,
    FinalAnswer,
    Finding,
    FindingDraft,
    Plan,
    Reason,
    RefinedQueries,
    RefinedQuery,
    RunStatus,
    SearchArgs,
    SearchResult,
    Source,
    Strategy,
    SubQuestion,
    SubQuestionCoverage,
    SynthesisDraft,
    ToolError,
    UsageSummary,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_source(**kwargs: object) -> Source:
    defaults: dict = {
        "id": "S1",
        "url": "https://example.com/article",
        "domain": "example.com",
        "title": "Example Article",
        "quality_score": 0.8,
        "fetched": True,
        "snippet_only": False,
    }
    defaults.update(kwargs)
    return Source(**defaults)  # type: ignore[arg-type]


def _make_finding(**kwargs: object) -> Finding:
    defaults: dict = {"statement": "The sky is blue.", "evidence_ids": ["E1"]}
    defaults.update(kwargs)
    return Finding(**defaults)  # type: ignore[arg-type]


def _make_conflict(**kwargs: object) -> Conflict:
    defaults: dict = {
        "topic": "color",
        "description": "Sources disagree on color.",
        "evidence_ids": ["E1", "E2"],
    }
    defaults.update(kwargs)
    return Conflict(**defaults)  # type: ignore[arg-type]


def _make_coverage() -> Coverage:
    return Coverage(
        total=2,
        covered=2,
        per_subquestion=[
            SubQuestionCoverage(id="q1", text="Q1", evidence_count=1, independent_domains=1),
            SubQuestionCoverage(id="q2", text="Q2", evidence_count=2, independent_domains=2),
        ],
    )


def _make_confidence_report(**kwargs: object) -> ConfidenceReport:
    defaults: dict = {
        "level": Confidence.HIGH,
        "coverage_ratio": 1.0,
        "avg_source_quality": 0.8,
        "min_independent_domains": 2,
        "unresolved_conflicts": 0,
        "snippet_only_ratio": 0.0,
        "verification_complete": True,
        "reasons": ["All sub-questions covered"],
    }
    defaults.update(kwargs)
    return ConfidenceReport(**defaults)  # type: ignore[arg-type]


def _make_usage() -> UsageSummary:
    return UsageSummary(
        llm_calls=5,
        input_tokens=1000,
        output_tokens=500,
        searches=3,
        fetches=6,
        tool_retries=0,
        latency_s=12.3,
    )


def _make_final_answer() -> FinalAnswer:
    source = _make_source(text="full text hidden")
    finding = _make_finding()
    conflict = _make_conflict()
    return FinalAnswer(
        run_id="run-001",
        user_question="What is the capital of France?",
        short_answer="Paris",
        key_findings=[finding],
        sources_used=[source],
        confidence=_make_confidence_report(),
        limitations_and_assumptions=["Limited to public data"],
        suggested_next_steps=["Visit Paris"],
        conflicts=[conflict],
        coverage=_make_coverage(),
        status=RunStatus.DONE,
        reasons=[],
        usage=_make_usage(),
    )


def _make_sub_question(id: str = "q1") -> SubQuestion:
    return SubQuestion(
        id=id,
        text="What is the capital?",
        queries=["capital of France"],
        strategy=Strategy.search_and_fetch,
        time_sensitive=False,
        rationale="Basic geography",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────


def test_strategy_values() -> None:
    assert Strategy.search_only == "search_only"
    assert Strategy.search_and_fetch == "search_and_fetch"


def test_error_class_values() -> None:
    for name in ("TRANSIENT", "CLIENT", "CONTENT", "SCHEMA", "SECURITY", "BUDGET", "UNKNOWN"):
        assert ErrorClass[name].value == name


def test_fetch_status_values() -> None:
    for name in ("OK", "EMPTY", "BLOCKED", "NOT_FOUND", "HTTP_ERROR", "NON_HTML",
                 "TOO_LARGE", "SECURITY_BLOCKED"):
        assert FetchStatus[name].value == name


def test_confidence_values() -> None:
    assert Confidence.HIGH == "HIGH"
    assert Confidence.MEDIUM == "MEDIUM"
    assert Confidence.LOW == "LOW"


def test_run_status_values() -> None:
    assert RunStatus.DONE == "DONE"
    assert RunStatus.DEGRADED == "DEGRADED"
    assert RunStatus.FAILED_GRACEFULLY == "FAILED_GRACEFULLY"


def test_reason_all_values() -> None:
    expected = {
        "UNCOVERED_SUBQUESTIONS", "BUDGET_EXHAUSTED", "SNIPPET_ONLY_EVIDENCE",
        "FINDINGS_DROPPED", "PROVIDER_FALLBACK", "VERIFICATION_UNAVAILABLE",
        "EXTRACTION_FAILED", "NODE_ERROR", "NOT_ANSWERABLE",
        "INSUFFICIENT_EVIDENCE", "ALL_FINDINGS_UNSUPPORTED", "PLAN_FALLBACK",
    }
    actual = {r.value for r in Reason}
    assert actual == expected


# ──────────────────────────────────────────────────────────────────────────────
# SearchArgs
# ──────────────────────────────────────────────────────────────────────────────


def test_search_args_strips_whitespace() -> None:
    sa = SearchArgs(query="  hello world  ")
    assert sa.query == "hello world"


def test_search_args_valid() -> None:
    sa = SearchArgs(query="What is Python?", max_results=3, depth="advanced")
    assert sa.query == "What is Python?"
    assert sa.max_results == 3
    assert sa.depth == "advanced"


def test_search_args_default_depth_and_results() -> None:
    sa = SearchArgs(query="foo bar")
    assert sa.depth == "basic"
    assert sa.max_results == 5


def test_search_args_too_short() -> None:
    with pytest.raises(ValueError):
        SearchArgs(query="ab")


def test_search_args_too_long() -> None:
    with pytest.raises(ValueError):
        SearchArgs(query="x" * 201)


def test_search_args_url_rejected() -> None:
    with pytest.raises(ValueError, match="URL"):
        SearchArgs(query="https://example.com/search")


def test_search_args_http_url_rejected() -> None:
    with pytest.raises(ValueError, match="URL"):
        SearchArgs(query="http://example.com")


def test_search_args_max_results_out_of_range() -> None:
    with pytest.raises(ValueError):
        SearchArgs(query="valid query", max_results=11)
    with pytest.raises(ValueError):
        SearchArgs(query="valid query", max_results=0)


def test_search_args_extra_field_rejected() -> None:
    with pytest.raises(ValueError):
        SearchArgs(query="valid query", unknown_field="x")  # type: ignore[call-arg]


# ──────────────────────────────────────────────────────────────────────────────
# SubQuestion
# ──────────────────────────────────────────────────────────────────────────────


def test_sub_question_valid_id() -> None:
    sq = SubQuestion(
        id="q1",
        text="What is X?",
        queries=["X query"],
        strategy=Strategy.search_only,
        time_sensitive=False,
        rationale="Basic lookup",
    )
    assert sq.id == "q1"


def test_sub_question_invalid_id() -> None:
    with pytest.raises(ValueError, match=r"\^q\[0-9\]\+\$"):
        SubQuestion(
            id="sq1",
            text="What?",
            queries=["query"],
            strategy=Strategy.search_only,
            time_sensitive=False,
            rationale="r",
        )


def test_sub_question_too_many_queries() -> None:
    with pytest.raises(ValueError):
        SubQuestion(
            id="q1",
            text="What?",
            queries=["q1", "q2", "q3"],  # 3 > max 2
            strategy=Strategy.search_only,
            time_sensitive=False,
            rationale="r",
        )


def test_sub_question_no_queries() -> None:
    with pytest.raises(ValueError):
        SubQuestion(
            id="q1",
            text="What?",
            queries=[],
            strategy=Strategy.search_only,
            time_sensitive=False,
            rationale="r",
        )


def test_sub_question_query_url_rejected() -> None:
    """URL-like string inside SubQuestion.queries must be rejected."""
    with pytest.raises(ValueError, match="URL"):
        SubQuestion(
            id="q1",
            text="What?",
            queries=["https://example.com/search"],
            strategy=Strategy.search_only,
            time_sensitive=False,
            rationale="rationale text",
        )


def test_sub_question_query_http_url_rejected() -> None:
    with pytest.raises(ValueError, match="URL"):
        SubQuestion(
            id="q1",
            text="What?",
            queries=["http://example.com"],
            strategy=Strategy.search_only,
            time_sensitive=False,
            rationale="rationale text",
        )


def test_sub_question_query_too_short_rejected() -> None:
    """A 2-character query inside SubQuestion.queries must be rejected."""
    with pytest.raises(ValueError):
        SubQuestion(
            id="q1",
            text="What?",
            queries=["ab"],
            strategy=Strategy.search_only,
            time_sensitive=False,
            rationale="rationale text",
        )


def test_sub_question_queries_stripped() -> None:
    """Queries inside SubQuestion.queries come back stripped."""
    sq = SubQuestion(
        id="q1",
        text="What?",
        queries=["  valid query  "],
        strategy=Strategy.search_only,
        time_sensitive=False,
        rationale="rationale text",
    )
    assert sq.queries == ["valid query"]


def test_sub_question_blank_text_rejected() -> None:
    """Blank SubQuestion.text must be rejected."""
    with pytest.raises(ValueError):
        SubQuestion(
            id="q1",
            text="   ",
            queries=["valid query"],
            strategy=Strategy.search_only,
            time_sensitive=False,
            rationale="rationale text",
        )


def test_sub_question_blank_rationale_rejected() -> None:
    """Blank SubQuestion.rationale must be rejected."""
    with pytest.raises(ValueError):
        SubQuestion(
            id="q1",
            text="What is X?",
            queries=["valid query"],
            strategy=Strategy.search_only,
            time_sensitive=False,
            rationale="   ",
        )


def test_sub_question_extra_field_rejected() -> None:
    with pytest.raises(ValueError):
        SubQuestion(  # type: ignore[call-arg]
            id="q1",
            text="What?",
            queries=["valid query"],
            strategy=Strategy.search_only,
            time_sensitive=False,
            rationale="r",
            unknown="bad",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Plan
# ──────────────────────────────────────────────────────────────────────────────


def test_plan_answerable_valid() -> None:
    p = Plan(
        restated_question="Capital of France?",
        is_answerable=True,
        assumptions=[],
        sub_questions=[_make_sub_question("q1")],
    )
    assert p.is_answerable is True


def test_plan_not_answerable_requires_clarification() -> None:
    with pytest.raises(ValueError, match="clarification_note"):
        Plan(
            restated_question="???",
            is_answerable=False,
            assumptions=[],
            sub_questions=[],
        )


def test_plan_not_answerable_valid() -> None:
    p = Plan(
        restated_question="???",
        is_answerable=False,
        clarification_note="Too vague",
        assumptions=[],
        sub_questions=[],
    )
    assert p.clarification_note == "Too vague"


def test_plan_answerable_no_sub_questions() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Plan(
            restated_question="Capital?",
            is_answerable=True,
            assumptions=[],
            sub_questions=[],
        )


def test_plan_duplicate_sub_question_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        Plan(
            restated_question="Capital?",
            is_answerable=True,
            assumptions=[],
            sub_questions=[_make_sub_question("q1"), _make_sub_question("q1")],
        )


def test_plan_six_sub_questions_accepted() -> None:
    """A plan with 6 sub-questions must be accepted.

    SPEC v3.1: the upper limit is enforced by the plan node via TRUNCATION, NOT by the schema.
    """
    p = Plan(
        restated_question="Big question?",
        is_answerable=True,
        assumptions=[],
        sub_questions=[_make_sub_question(f"q{i}") for i in range(1, 7)],
    )
    assert len(p.sub_questions) == 6


def test_plan_more_than_six_sub_questions_accepted() -> None:
    """A plan with more than 5 sub-questions must be accepted by the schema."""
    p = Plan(
        restated_question="Big question?",
        is_answerable=True,
        assumptions=[],
        sub_questions=[_make_sub_question(f"q{i}") for i in range(1, 9)],
    )
    assert len(p.sub_questions) == 8


def test_plan_not_answerable_whitespace_only_note_rejected() -> None:
    """A non-answerable plan with a whitespace-only clarification_note must be rejected."""
    with pytest.raises(ValueError, match="clarification_note"):
        Plan(
            restated_question="???",
            is_answerable=False,
            clarification_note="   ",
            assumptions=[],
            sub_questions=[],
        )


def test_plan_not_answerable_empty_string_note_rejected() -> None:
    """A non-answerable plan with an empty string clarification_note must be rejected."""
    with pytest.raises(ValueError, match="clarification_note"):
        Plan(
            restated_question="???",
            is_answerable=False,
            clarification_note="",
            assumptions=[],
            sub_questions=[],
        )


def test_plan_extra_field_rejected() -> None:
    with pytest.raises(ValueError):
        Plan(  # type: ignore[call-arg]
            restated_question="Q",
            is_answerable=False,
            clarification_note="note",
            assumptions=[],
            sub_questions=[],
            extra_field="bad",
        )


# ──────────────────────────────────────────────────────────────────────────────
# ToolError & retriable property
# ──────────────────────────────────────────────────────────────────────────────


def test_tool_error_retriable_transient() -> None:
    te = ToolError(tool="search", error_class=ErrorClass.TRANSIENT, message="timeout")
    assert te.retriable is True


def test_tool_error_not_retriable_client() -> None:
    te = ToolError(tool="search", error_class=ErrorClass.CLIENT, message="bad key")
    assert te.retriable is False


def test_tool_error_not_retriable_schema() -> None:
    te = ToolError(tool="llm", error_class=ErrorClass.SCHEMA, message="bad json")
    assert te.retriable is False


def test_tool_error_retriable_is_property_not_field() -> None:
    assert "retriable" not in ToolError.model_fields


def test_tool_error_extra_field_rejected() -> None:
    with pytest.raises(ValueError):
        ToolError(  # type: ignore[call-arg]
            tool="search", error_class=ErrorClass.TRANSIENT, message="t", unknown="x"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Source — text excluded from serialization and repr
# ──────────────────────────────────────────────────────────────────────────────


def test_source_text_excluded_from_json() -> None:
    s = _make_source(text="this is full page text that must be hidden")
    j = s.model_dump_json()
    assert "full page text" not in j
    assert "text" not in j  # the key itself must not appear


def test_source_text_excluded_from_dict() -> None:
    s = _make_source(text="hidden content")
    d = s.model_dump()
    assert "text" not in d


def test_source_text_excluded_from_repr() -> None:
    s = _make_source(text="secret page contents")
    r = repr(s)
    assert "secret page contents" not in r
    # The field name 'text' may appear (other fields share the substring),
    # but the VALUE must not.


def test_source_quality_score_out_of_range() -> None:
    with pytest.raises(ValueError):
        _make_source(quality_score=1.5)
    with pytest.raises(ValueError):
        _make_source(quality_score=-0.1)


def test_source_quality_score_boundary_valid() -> None:
    s0 = _make_source(quality_score=0.0)
    s1 = _make_source(quality_score=1.0)
    assert s0.quality_score == 0.0
    assert s1.quality_score == 1.0


def test_source_extra_field_rejected() -> None:
    with pytest.raises(ValueError):
        _make_source(unknown_field="bad")  # type: ignore[call-arg]


# ──────────────────────────────────────────────────────────────────────────────
# EvidenceItem
# ──────────────────────────────────────────────────────────────────────────────


def test_evidence_item_valid() -> None:
    e = EvidenceItem(
        id="E1",
        sub_question_id="q1",
        source_id="S1",
        domain="example.com",
        passage="The sky is blue.",
        claim="Sky is blue.",
        snippet_only=False,
    )
    assert e.id == "E1"


def test_evidence_item_empty_passage_rejected() -> None:
    with pytest.raises(ValueError, match="passage"):
        EvidenceItem(
            id="E1",
            sub_question_id="q1",
            source_id="S1",
            domain="example.com",
            passage="   ",
            claim="claim",
            snippet_only=False,
        )


def test_evidence_item_extra_field_rejected() -> None:
    with pytest.raises(ValueError):
        EvidenceItem(  # type: ignore[call-arg]
            id="E1",
            sub_question_id="q1",
            source_id="S1",
            domain="example.com",
            passage="The sky is blue.",
            claim="Blue sky.",
            snippet_only=False,
            unknown="bad",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Conflict
# ──────────────────────────────────────────────────────────────────────────────


def test_conflict_valid() -> None:
    c = _make_conflict()
    assert len(c.evidence_ids) == 2


def test_conflict_single_evidence_id_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        Conflict(topic="t", description="d", evidence_ids=["E1"])


def test_conflict_duplicate_evidence_ids_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        Conflict(topic="t", description="d", evidence_ids=["E1", "E1"])


def test_conflict_extra_field_rejected() -> None:
    with pytest.raises(ValueError):
        Conflict(topic="t", description="d", evidence_ids=["E1", "E2"], extra="bad")  # type: ignore[call-arg]


# ──────────────────────────────────────────────────────────────────────────────
# FindingDraft
# ──────────────────────────────────────────────────────────────────────────────


def test_finding_draft_valid() -> None:
    fd = FindingDraft(statement="stmt", evidence_ids=["E1", "E2"])
    assert fd.evidence_ids == ["E1", "E2"]


def test_finding_draft_empty_evidence_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        FindingDraft(statement="stmt", evidence_ids=[])


def test_finding_draft_duplicate_evidence_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        FindingDraft(statement="stmt", evidence_ids=["E1", "E1"])


# ──────────────────────────────────────────────────────────────────────────────
# Finding
# ──────────────────────────────────────────────────────────────────────────────


def test_finding_default_support() -> None:
    f = _make_finding()
    assert f.support == "unchecked"


def test_finding_supported() -> None:
    f = Finding(statement="stmt", evidence_ids=["E1"], support="supported")
    assert f.support == "supported"


def test_finding_empty_evidence_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        Finding(statement="stmt", evidence_ids=[])


def test_finding_duplicate_evidence_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        Finding(statement="stmt", evidence_ids=["E1", "E1"])


def test_finding_extra_field_rejected() -> None:
    with pytest.raises(ValueError):
        Finding(statement="stmt", evidence_ids=["E1"], extra="bad")  # type: ignore[call-arg]


# ──────────────────────────────────────────────────────────────────────────────
# ExtractionOutput / EntailmentOutput / RefinedQueries (LLM-facing)
# ──────────────────────────────────────────────────────────────────────────────


def test_extraction_output_extra_rejected() -> None:
    with pytest.raises(ValueError):
        ExtractionOutput(items=[], extra_field="bad")  # type: ignore[call-arg]


def test_entailment_output_valid() -> None:
    eo = EntailmentOutput(
        items=[EntailmentItem(index=0, verdict="supported", reason="r")]
    )
    assert len(eo.items) == 1


def test_entailment_output_extra_rejected() -> None:
    with pytest.raises(ValueError):
        EntailmentOutput(items=[], extra="bad")  # type: ignore[call-arg]


def test_refined_queries_valid() -> None:
    rq = RefinedQueries(
        items=[RefinedQuery(sub_question_id="q1", queries=["new query"])]
    )
    assert len(rq.items) == 1


def test_refined_query_too_many_queries() -> None:
    with pytest.raises(ValueError):
        RefinedQuery(sub_question_id="q1", queries=["q1", "q2", "q3"])


def test_refined_queries_extra_rejected() -> None:
    with pytest.raises(ValueError):
        RefinedQueries(items=[], bad="x")  # type: ignore[call-arg]


def test_refined_query_url_rejected() -> None:
    """URL-like string inside RefinedQuery.queries must be rejected."""
    with pytest.raises(ValueError, match="URL"):
        RefinedQuery(sub_question_id="q1", queries=["https://example.com"])


def test_refined_query_http_url_rejected() -> None:
    with pytest.raises(ValueError, match="URL"):
        RefinedQuery(sub_question_id="q1", queries=["http://example.com"])


def test_refined_query_too_short_rejected() -> None:
    """A 2-character query inside RefinedQuery.queries must be rejected."""
    with pytest.raises(ValueError):
        RefinedQuery(sub_question_id="q1", queries=["ab"])


def test_refined_query_queries_stripped() -> None:
    """Queries inside RefinedQuery.queries come back stripped."""
    rq = RefinedQuery(sub_question_id="q1", queries=["  stripped query  "])
    assert rq.queries == ["stripped query"]


# ──────────────────────────────────────────────────────────────────────────────
# Coverage / SubQuestionCoverage — extra fields rejected
# ──────────────────────────────────────────────────────────────────────────────


def test_sub_question_coverage_extra_field_rejected() -> None:
    with pytest.raises(ValueError):
        SubQuestionCoverage(  # type: ignore[call-arg]
            id="q1", text="Q1", evidence_count=1, independent_domains=1, bad="x"
        )


def test_coverage_extra_field_rejected() -> None:
    with pytest.raises(ValueError):
        Coverage(  # type: ignore[call-arg]
            total=1, covered=1, per_subquestion=[], bad="x"
        )


# ──────────────────────────────────────────────────────────────────────────────
# ConfidenceReport — range constraints
# ──────────────────────────────────────────────────────────────────────────────


def test_confidence_report_valid() -> None:
    cr = _make_confidence_report()
    assert cr.coverage_ratio == 1.0


def test_confidence_report_coverage_ratio_above_one_rejected() -> None:
    with pytest.raises(ValueError):
        _make_confidence_report(coverage_ratio=1.01)


def test_confidence_report_coverage_ratio_below_zero_rejected() -> None:
    with pytest.raises(ValueError):
        _make_confidence_report(coverage_ratio=-0.01)


def test_confidence_report_avg_source_quality_above_one_rejected() -> None:
    with pytest.raises(ValueError):
        _make_confidence_report(avg_source_quality=1.5)


def test_confidence_report_avg_source_quality_below_zero_rejected() -> None:
    with pytest.raises(ValueError):
        _make_confidence_report(avg_source_quality=-0.1)


def test_confidence_report_snippet_only_ratio_above_one_rejected() -> None:
    with pytest.raises(ValueError):
        _make_confidence_report(snippet_only_ratio=1.1)


def test_confidence_report_snippet_only_ratio_below_zero_rejected() -> None:
    with pytest.raises(ValueError):
        _make_confidence_report(snippet_only_ratio=-0.5)


def test_confidence_report_min_independent_domains_negative_rejected() -> None:
    with pytest.raises(ValueError):
        _make_confidence_report(min_independent_domains=-1)


def test_confidence_report_unresolved_conflicts_negative_rejected() -> None:
    with pytest.raises(ValueError):
        _make_confidence_report(unresolved_conflicts=-1)


def test_confidence_report_boundary_values_valid() -> None:
    """Boundary values 0.0 and 1.0 must be accepted."""
    cr = _make_confidence_report(
        coverage_ratio=0.0,
        avg_source_quality=0.0,
        snippet_only_ratio=1.0,
        min_independent_domains=0,
        unresolved_conflicts=0,
    )
    assert cr.coverage_ratio == 0.0
    assert cr.snippet_only_ratio == 1.0


def test_confidence_report_extra_field_rejected() -> None:
    with pytest.raises(ValueError):
        _make_confidence_report(unknown="bad")  # type: ignore[call-arg]


# ──────────────────────────────────────────────────────────────────────────────
# UsageSummary — extra fields rejected
# ──────────────────────────────────────────────────────────────────────────────


def test_usage_summary_no_cost() -> None:
    u = _make_usage()
    assert u.est_cost_usd is None


def test_usage_summary_with_cost() -> None:
    u = UsageSummary(
        llm_calls=1, input_tokens=100, output_tokens=50,
        searches=1, fetches=1, tool_retries=0, latency_s=1.0,
        est_cost_usd=0.001,
    )
    assert u.est_cost_usd == pytest.approx(0.001)


def test_usage_summary_extra_field_rejected() -> None:
    with pytest.raises(ValueError):
        UsageSummary(  # type: ignore[call-arg]
            llm_calls=1, input_tokens=100, output_tokens=50,
            searches=1, fetches=1, tool_retries=0, latency_s=1.0,
            surprise="bad",
        )


# ──────────────────────────────────────────────────────────────────────────────
# FinalAnswer round-trip
# ──────────────────────────────────────────────────────────────────────────────


def test_final_answer_round_trip() -> None:
    fa = _make_final_answer()
    json_str = fa.model_dump_json()
    fa2 = FinalAnswer.model_validate_json(json_str)
    assert fa2.run_id == fa.run_id
    assert fa2.short_answer == fa.short_answer
    assert fa2.status == fa.status
    assert len(fa2.key_findings) == len(fa.key_findings)
    assert len(fa2.sources_used) == len(fa.sources_used)


def test_final_answer_source_text_not_in_json() -> None:
    fa = _make_final_answer()
    j = fa.model_dump_json()
    # Source.text is excluded; its value must not leak
    assert "full text hidden" not in j


def test_final_answer_extra_field_rejected() -> None:
    fa = _make_final_answer()
    d = fa.model_dump()
    d["surprise_field"] = "oops"
    with pytest.raises(ValueError):
        FinalAnswer(**d)


# ──────────────────────────────────────────────────────────────────────────────
# JSON schema generation (must not raise)
# SPEC requires: Plan, ExtractionOutput, SynthesisDraft, EntailmentOutput,
#                RefinedQueries, FinalAnswer
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model_cls",
    [
        Plan,
        ExtractionOutput,
        SynthesisDraft,
        EntailmentOutput,
        RefinedQueries,
        FinalAnswer,
    ],
)
def test_json_schema_generates(model_cls: type) -> None:  # type: ignore[type-arg]
    schema = model_cls.model_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema or "title" in schema


# ──────────────────────────────────────────────────────────────────────────────
# AgentState
# ──────────────────────────────────────────────────────────────────────────────


def test_agent_state_defaults() -> None:
    state = AgentState(question="What?", today="2024-01-01")
    assert state.plan is None
    assert state.search_results == {}
    assert state.sources == {}
    assert state.evidence == []
    assert state.pending_sq_ids == []
    assert state.refine_loops == 0
    assert state.draft is None
    assert state.findings == []
    assert state.conflicts == []
    assert state.limitations == []
    assert state.next_steps == []
    assert state.short_answer is None
    assert state.final is None
    assert state.reasons == []
    assert state.abort is False
    assert state.errors == []
    assert state.stats == {}


def test_agent_state_route_defaults_to_none() -> None:
    """AgentState.route must default to None."""
    state = AgentState(question="q", today="2026-01-01")
    assert state.route is None


def test_agent_state_route_valid_values() -> None:
    """AgentState.route accepts the three valid literal values."""
    for val in ("refine", "synthesize", "insufficient"):
        state = AgentState(question="q", today="2026-01-01", route=val)  # type: ignore[arg-type]
        assert state.route == val


def test_agent_state_route_invalid_value_rejected() -> None:
    """An invalid route value must be rejected."""
    with pytest.raises(ValueError):
        AgentState(question="q", today="2026-01-01", route="unknown_route")  # type: ignore[arg-type]


def test_agent_state_extra_field_rejected() -> None:
    with pytest.raises(ValueError):
        AgentState(question="q", today="2024-01-01", surprise="x")  # type: ignore[call-arg]


# ──────────────────────────────────────────────────────────────────────────────
# SynthesisDraft (LLM-facing) — extra fields rejected
# ──────────────────────────────────────────────────────────────────────────────


def test_synthesis_draft_extra_field_rejected() -> None:
    with pytest.raises(ValueError):
        SynthesisDraft(  # type: ignore[call-arg]
            short_answer="ans",
            findings=[],
            conflicts=[],
            limitations=[],
            next_steps=[],
            extra="bad",
        )


def test_synthesis_draft_valid() -> None:
    sd = SynthesisDraft(
        short_answer="Answer here",
        findings=[FindingDraft(statement="s", evidence_ids=["E1"])],
        conflicts=[],
        limitations=["one"],
        next_steps=["do something"],
    )
    assert sd.short_answer == "Answer here"


# ──────────────────────────────────────────────────────────────────────────────
# FetchResult — stays permissive (unknown fields accepted)
# ──────────────────────────────────────────────────────────────────────────────


def test_fetch_result_ok() -> None:
    fr = FetchResult(url="https://example.com", status=FetchStatus.OK, text="content")
    assert fr.status == FetchStatus.OK
    assert fr.truncated is False


def test_fetch_result_with_error() -> None:
    err = ToolError(tool="fetch", error_class=ErrorClass.TRANSIENT, message="timeout")
    fr = FetchResult(url="https://example.com", status=FetchStatus.HTTP_ERROR, error=err)
    assert fr.error is not None
    assert fr.error.retriable is True


def test_fetch_result_accepts_unknown_fields() -> None:
    """FetchResult must stay permissive and accept unknown fields."""
    fr = FetchResult(
        url="https://example.com",
        status=FetchStatus.OK,
        extra_provider_field="some_value",  # type: ignore[call-arg]
    )
    assert fr.status == FetchStatus.OK


# ──────────────────────────────────────────────────────────────────────────────
# SearchResult — stays permissive (unknown fields accepted)
# ──────────────────────────────────────────────────────────────────────────────


def test_search_result_valid() -> None:
    sr = SearchResult(
        url="https://example.com",
        title="Example",
        snippet="Some text",
        provider="tavily",
    )
    assert sr.score is None
    assert sr.sub_question_id is None


def test_search_result_accepts_unknown_fields() -> None:
    """SearchResult must stay permissive and accept unknown fields."""
    sr = SearchResult(
        url="https://example.com",
        title="Example",
        snippet="Some text",
        provider="tavily",
        extra_provider_field="foo",  # type: ignore[call-arg]
    )
    assert sr.title == "Example"


# ──────────────────────────────────────────────────────────────────────────────
# ExtractionItem (LLM-facing)
# ──────────────────────────────────────────────────────────────────────────────


def test_extraction_item_extra_rejected() -> None:
    with pytest.raises(ValueError):
        ExtractionItem(source_id="S1", passage="p", claim="c", extra="bad")  # type: ignore[call-arg]


def test_extraction_item_valid() -> None:
    ei = ExtractionItem(source_id="S1", passage="The sky is blue", claim="Blue sky")
    assert ei.source_id == "S1"


# ──────────────────────────────────────────────────────────────────────────────
# JSON serialization: validate no dict fields in LLM-facing models
# ──────────────────────────────────────────────────────────────────────────────


def test_plan_schema_no_dict_types() -> None:
    """LLM-facing models must not use dict fields (SPEC section 4)."""
    schema = Plan.model_json_schema()
    schema_str = json.dumps(schema)
    # 'object' with 'additionalProperties' would indicate a dict field
    # AgentState (not LLM-facing) has dicts, Plan must not
    props = schema.get("properties", {})
    for field_name, field_schema in props.items():
        assert field_schema.get("type") != "object" or field_name in (
            # none expected in LLM-facing Plan
        ), f"Plan has unexpected dict-type field: {field_name!r}"
    # Just ensure schema str is well-formed JSON
    assert json.loads(schema_str)
