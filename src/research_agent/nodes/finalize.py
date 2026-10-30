"""Build, validate, and persist the final answer without raising."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

from research_agent.config import Settings
from research_agent.deps import Deps, RunContext
from research_agent.nodes.sufficiency import compute_coverage
from research_agent.schemas import (
    AgentState,
    Confidence,
    ConfidenceReport,
    Conflict,
    Coverage,
    EvidenceItem,
    FinalAnswer,
    Finding,
    Reason,
    RunStatus,
    Source,
    UsageSummary,
)

_LOGGER = logging.getLogger(__name__)


def compute_status(
    reasons: list[Reason],
    *,
    answerable: bool,
    has_findings: bool,
) -> RunStatus:
    """Derive run status solely from the SPEC's answerability and reason rules."""
    failed_reasons = {
        Reason.NOT_ANSWERABLE,
        Reason.INSUFFICIENT_EVIDENCE,
        Reason.ALL_FINDINGS_UNSUPPORTED,
    }
    if not answerable or not has_findings or any(reason in failed_reasons for reason in reasons):
        return RunStatus.FAILED_GRACEFULLY
    if reasons:
        return RunStatus.DEGRADED
    return RunStatus.DONE


def compute_confidence(
    *,
    coverage: Coverage,
    findings: list[Finding],
    sources: list[Source],
    conflicts_count: int,
    verification_complete: bool,
    status: RunStatus,
    settings: Settings,
) -> ConfidenceReport:
    """Calculate rule-based confidence and plain-language threshold explanations."""
    ratio = coverage.covered / coverage.total if coverage.total else 0.0
    avg_quality = mean(source.quality_score for source in sources) if sources else 0.0
    snippet_ratio = (
        sum(source.snippet_only for source in sources) / len(sources) if sources else 0.0
    )
    covered_questions = [
        item for item in coverage.per_subquestion if item.evidence_count > 0
    ]
    min_domains = min(
        (item.independent_domains for item in covered_questions),
        default=0,
    )
    high_checks = {
        "coverage is complete": ratio == 1.0,
        f"each covered sub-question has at least {settings.HIGH_MIN_DOMAINS} independent domains":
            bool(covered_questions)
            and all(
                item.independent_domains >= settings.HIGH_MIN_DOMAINS
                for item in covered_questions
            ),
        f"average source quality meets {settings.HIGH_MIN_AVG_QUALITY:.2f}":
            avg_quality >= settings.HIGH_MIN_AVG_QUALITY,
        "there are no unresolved conflicts": conflicts_count == 0,
        f"snippet-only share is at most {settings.HIGH_MAX_SNIPPET_RATIO:.2f}":
            snippet_ratio <= settings.HIGH_MAX_SNIPPET_RATIO,
        "verification is complete": verification_complete,
        "run status is DONE": status is RunStatus.DONE,
    }
    medium_checks = {
        f"coverage meets {settings.MEDIUM_MIN_COVERAGE:.2f}":
            ratio >= settings.MEDIUM_MIN_COVERAGE,
        f"average source quality meets {settings.MEDIUM_MIN_AVG_QUALITY:.2f}":
            avg_quality >= settings.MEDIUM_MIN_AVG_QUALITY,
        f"unresolved conflicts are at most {settings.MEDIUM_MAX_CONFLICTS}":
            conflicts_count <= settings.MEDIUM_MAX_CONFLICTS,
    }

    if status is RunStatus.FAILED_GRACEFULLY or not findings:
        level = Confidence.LOW
        reasons = [
            "Run status is FAILED_GRACEFULLY."
            if status is RunStatus.FAILED_GRACEFULLY
            else "There are no verified findings."
        ]
    elif all(high_checks.values()):
        level = Confidence.HIGH
        reasons = [
            "All HIGH confidence conditions are satisfied.",
            *(f"Pass: {name}." for name in high_checks),
        ]
    elif all(medium_checks.values()):
        level = Confidence.MEDIUM
        reasons = [
            "All MEDIUM confidence conditions are satisfied.",
            *(f"Pass: {name}." for name in medium_checks),
            *(f"Not met for HIGH: {name}." for name, passed in high_checks.items() if not passed),
        ]
    else:
        level = Confidence.LOW
        reasons = [
            "Neither the HIGH nor MEDIUM confidence conditions are all satisfied.",
            *(
                f"Not met for MEDIUM: {name}."
                for name, passed in medium_checks.items()
                if not passed
            ),
        ]
    if not verification_complete and level is Confidence.MEDIUM:
        reasons.append("Incomplete verification caps confidence at MEDIUM.")

    return ConfidenceReport(
        level=level,
        coverage_ratio=ratio,
        avg_source_quality=avg_quality,
        min_independent_domains=min_domains,
        unresolved_conflicts=conflicts_count,
        snippet_only_ratio=snippet_ratio,
        verification_complete=verification_complete,
        reasons=reasons,
    )


def _sources_in_citation_order(
    findings: list[Finding],
    conflicts: list[Conflict],
    evidence: list[EvidenceItem],
    sources: dict[str, Source],
) -> list[Source]:
    source_by_evidence = {item.id: item.source_id for item in evidence}
    ordered_ids: list[str] = []
    for citations in (
        *(finding.evidence_ids for finding in findings),
        *(conflict.evidence_ids for conflict in conflicts),
    ):
        for evidence_id in citations:
            source_id = source_by_evidence.get(evidence_id)
            if source_id is not None and source_id in sources and source_id not in ordered_ids:
                ordered_ids.append(source_id)
    return [sources[source_id] for source_id in ordered_ids]


def _auto_limitations(
    state: AgentState,
    coverage: Coverage,
    sources_used: list[Source],
) -> list[str]:
    limitations = list(state.limitations)
    for question in coverage.per_subquestion:
        if question.evidence_count == 0:
            text = f"No verified evidence was found for: {question.text}"
            if text not in limitations:
                limitations.append(text)

    if sources_used:
        snippet_ratio = sum(item.snippet_only for item in sources_used) / len(sources_used)
        if snippet_ratio > 0:
            text = f"{snippet_ratio:.0%} of evidence items came from snippets only."
            if text not in limitations:
                limitations.append(text)

    time_sensitive_ids = {
        question.id
        for question in state.plan.sub_questions
        if question.time_sensitive
    } if state.plan else set()
    stale_sources = {
        item.source_id
        for item in state.evidence
        if item.sub_question_id in time_sensitive_ids
        and item.source_id in state.sources
        and "stale" in state.sources[item.source_id].flags
    }
    if any(source.id in stale_sources for source in sources_used):
        limitations.append("Some cited sources are stale for time-sensitive questions.")

    if any("possible_injection" in source.flags for source in sources_used):
        limitations.append(
            "Some cited sources contained instruction-like content that was removed."
        )
    if any(finding.support == "partial" for finding in state.findings):
        limitations.append("Some findings are only partially supported by their cited evidence.")
    if Reason.PROVIDER_FALLBACK in state.reasons:
        limitations.append("A search provider fallback was required.")
    if Reason.BUDGET_EXHAUSTED in state.reasons:
        limitations.append("A resource budget was exhausted before all research could complete.")
    if state.plan is not None:
        limitations.extend(state.plan.assumptions)
    return list(dict.fromkeys(limitations))


def _usage_summary(settings: Settings, run: RunContext) -> UsageSummary:
    snapshot = run.budget.snapshot()
    rates = (
        (settings.PRICE_IN_PER_MTOK_CHEAP, settings.PRICE_OUT_PER_MTOK_CHEAP),
        (settings.PRICE_IN_PER_MTOK_STRONG, settings.PRICE_OUT_PER_MTOK_STRONG),
    )
    if all(rate is None for pair in rates for rate in pair):
        estimated_cost = None
    else:
        estimated_cost = sum(
            float(details["cost_usd"])
            for event in run.tracer.events
            if isinstance((details := event.get("details")), dict)
            and isinstance(details.get("cost_usd"), (int, float))
        )
    return UsageSummary(
        llm_calls=snapshot["llm_calls"],
        input_tokens=snapshot["input_tokens"],
        output_tokens=snapshot["output_tokens"],
        searches=snapshot["searches"],
        fetches=snapshot["fetches"],
        tool_retries=snapshot["tool_retries"],
        latency_s=snapshot["elapsed_s"],
        est_cost_usd=estimated_cost,
    )


def _minimal_answer(
    state: AgentState,
    settings: Settings,
    run: RunContext,
) -> FinalAnswer:
    reasons = list(state.reasons)
    if Reason.NODE_ERROR not in reasons:
        reasons.append(Reason.NODE_ERROR)
    coverage = Coverage(total=0, covered=0, per_subquestion=[])
    confidence = ConfidenceReport(
        level=Confidence.LOW,
        coverage_ratio=0,
        avg_source_quality=0,
        min_independent_domains=0,
        unresolved_conflicts=0,
        snippet_only_ratio=0,
        verification_complete=False,
        reasons=["Finalization failed; confidence is LOW."],
    )
    return FinalAnswer(
        run_id=run.run_id,
        user_question=state.question,
        short_answer="Reliable information was not found due to an internal error.",
        key_findings=[],
        sources_used=[],
        confidence=confidence,
        limitations_and_assumptions=["Finalization encountered an internal error."],
        suggested_next_steps=["Retry the research after resolving the internal error."],
        conflicts=[],
        coverage=coverage,
        status=RunStatus.FAILED_GRACEFULLY,
        reasons=reasons,
        usage=_usage_summary(settings, run),
    )


def make_finalize_node(
    deps: Deps,
    run: RunContext,
) -> Callable[[AgentState], dict[str, Any]]:
    """Build the finalizer; all internal failures become a valid graceful answer."""

    def finalize_node(state: AgentState) -> dict[str, Any]:
        try:
            coverage = (
                compute_coverage(state.plan, state.evidence)
                if state.plan is not None
                else Coverage(total=0, covered=0, per_subquestion=[])
            )
            sources_used = _sources_in_citation_order(
                state.findings,
                state.conflicts,
                state.evidence,
                state.sources,
            )
            answerable = state.plan is not None and state.plan.is_answerable
            findings = list(state.findings)
            insufficient = (
                not answerable
                or Reason.INSUFFICIENT_EVIDENCE in state.reasons
                or Reason.ALL_FINDINGS_UNSUPPORTED in state.reasons
                or state.abort and not state.draft
            )
            if insufficient:
                findings = []
                if state.plan is not None and not state.plan.is_answerable:
                    short_answer = state.plan.clarification_note or (
                        "Reliable information was not found for the question."
                    )
                    clarification_limitation = (
                        f"Clarification is needed: {state.plan.clarification_note}"
                    )
                else:
                    short_answer = (
                        "Reliable information was not found for the requested sub-questions."
                    )
                    clarification_limitation = None
                next_steps = [
                    f"Search credible sources for: {item.text}"
                    for item in coverage.per_subquestion
                    if item.evidence_count == 0
                ]
                if not next_steps:
                    next_steps = [
                        "Provide a more specific question or narrower scope.",
                        "Consult primary or official sources for the missing information.",
                    ]
            else:
                short_answer = state.short_answer or (
                    "Reliable information was not found in the verified findings."
                )
                next_steps = list(state.next_steps)

            status = compute_status(
                state.reasons,
                answerable=answerable,
                has_findings=bool(findings),
            )
            confidence = compute_confidence(
                coverage=coverage,
                findings=findings,
                sources=sources_used,
                conflicts_count=len(state.conflicts),
                verification_complete=bool(state.stats.get("verification_complete", False)),
                status=status,
                settings=deps.settings,
            )
            limitations = _auto_limitations(state, coverage, sources_used)
            if insufficient and clarification_limitation:
                limitations.append(clarification_limitation)
            if not limitations and (
                status is RunStatus.FAILED_GRACEFULLY
                or Reason.INSUFFICIENT_EVIDENCE in state.reasons
            ):
                if Reason.PLAN_FALLBACK in state.reasons:
                    cause = (
                        state.errors[0].removeprefix("Planning step failed: ")
                        if state.errors
                        else "the planner output remained invalid after corrections"
                    )
                    limitations.append(f"The planning step failed: {cause}")
                elif state.errors:
                    limitations.append(
                        f"The research run could not complete: {state.errors[0]}"
                    )
                else:
                    limitations.append(
                        "Reliable evidence was insufficient to answer the question."
                    )
            answer = FinalAnswer(
                run_id=run.run_id,
                user_question=state.question,
                short_answer=short_answer,
                key_findings=findings,
                sources_used=sources_used,
                confidence=confidence,
                limitations_and_assumptions=limitations,
                suggested_next_steps=next_steps,
                conflicts=list(state.conflicts),
                coverage=coverage,
                status=status,
                reasons=list(state.reasons),
                usage=_usage_summary(deps.settings, run),
            )
            answer = FinalAnswer.model_validate(answer.model_dump(mode="python"))
        except Exception as error:
            answer = _minimal_answer(state, deps.settings, run)
            try:
                run.tracer.event(
                    "note",
                    "finalize",
                    details={
                        "action": "fallback_answer",
                        "error_type": type(error).__name__,
                    },
                )
            except Exception:
                pass

        try:
            run.tracer.event(
                "note",
                "finalize",
                details={"action": "finalized", "trace_summary": run.tracer.summary()},
            )
        except Exception as error:
            answer = _minimal_answer(state, deps.settings, run)
            _LOGGER.error("Finalization trace note failed (%s)", type(error).__name__)
        try:
            output_path = Path(deps.settings.RUNS_DIR) / f"{run.run_id}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(answer.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as error:
            answer = _minimal_answer(state, deps.settings, run)
            _LOGGER.error("Final answer persistence failed (%s)", type(error).__name__)
        return {"final": answer}

    return finalize_node
