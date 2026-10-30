"""Evidence-only strong-model synthesis node."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from research_agent.deps import Deps, RunContext
from research_agent.llm import ResilientLLM
from research_agent.prompts.synthesize import (
    synthesize_system_prompt,
    synthesize_user_prompt,
)
from research_agent.schemas import (
    AgentState,
    EvidenceItem,
    Finding,
    Reason,
    SynthesisDraft,
)


def _select_evidence(
    state: AgentState,
    max_evidence: int,
) -> list[EvidenceItem]:
    """Keep the highest-quality evidence fairly across questions, then restore order."""
    if max_evidence < 0:
        raise ValueError("max_evidence must not be negative")
    evidence = state.evidence
    if len(evidence) <= max_evidence:
        return evidence

    question_order = {
        question.id: index
        for index, question in enumerate(state.plan.sub_questions if state.plan else [])
    }
    grouped: dict[str, list[tuple[int, EvidenceItem]]] = defaultdict(list)
    for index, item in enumerate(evidence):
        grouped[item.sub_question_id].append((index, item))

    def quality(indexed_item: tuple[int, EvidenceItem]) -> tuple[float, int]:
        index, item = indexed_item
        source = state.sources.get(item.source_id)
        return (-(source.quality_score if source is not None else 0.0), index)

    ordered_question_ids = sorted(
        grouped,
        key=lambda question_id: (
            question_order.get(question_id, len(question_order)),
            question_id,
        ),
    )
    ranked_groups = {
        question_id: sorted(grouped[question_id], key=quality)
        for question_id in ordered_question_ids
    }
    selected_indexes: set[int] = set()
    rank = 0
    while len(selected_indexes) < max_evidence:
        added = False
        for question_id in ordered_question_ids:
            ranked = ranked_groups[question_id]
            if rank < len(ranked):
                selected_indexes.add(ranked[rank][0])
                added = True
                if len(selected_indexes) >= max_evidence:
                    break
        if not added:
            break
        rank += 1
    return [item for index, item in enumerate(evidence) if index in selected_indexes]


def make_synthesize_node(
    deps: Deps,
    run: RunContext,
) -> Callable[[AgentState], dict[str, Any]]:
    """Build the strong-model synthesis node with evidence-id validation."""
    llm = ResilientLLM(deps.llm, deps.settings, run.budget, run.tracer)

    def synthesize_node(state: AgentState) -> dict[str, Any]:
        with run.tracer.node_span("synthesize"):
            if state.plan is None:
                raise ValueError("synthesis requires a research plan")
            selected_evidence = _select_evidence(
                state,
                deps.settings.MAX_EVIDENCE_FOR_SYNTH,
            )
            evidence_by_id = {item.id: item for item in selected_evidence}
            valid_evidence_ids = frozenset(evidence_by_id)
            time_sensitive = any(
                question.time_sensitive for question in state.plan.sub_questions
            )
            user = synthesize_user_prompt(
                state.question,
                state.today,
                (
                    (question.id, question.text)
                    for question in state.plan.sub_questions
                ),
                (
                    {
                        "id": item.id,
                        "sub_question_id": item.sub_question_id,
                        "domain": item.domain,
                        "published_date": item.published_date,
                        "passage": item.passage,
                        "claim": item.claim,
                    }
                    for item in selected_evidence
                ),
            )

            def validate_draft(draft: SynthesisDraft) -> None:
                if not draft.findings:
                    raise ValueError("synthesis must include at least one finding")
                for finding in draft.findings:
                    if not finding.evidence_ids:
                        raise ValueError("every finding must cite evidence")
                    unknown = set(finding.evidence_ids) - valid_evidence_ids
                    if unknown:
                        raise ValueError(
                            f"finding cites unknown evidence ids: {sorted(unknown)}"
                        )
                for conflict in draft.conflicts:
                    unknown = set(conflict.evidence_ids) - valid_evidence_ids
                    if unknown:
                        raise ValueError(
                            "conflict cites unknown evidence ids: "
                            f"{sorted(unknown)}; each conflict must cite at least "
                            "two valid evidence ids"
                        )
                    if len(conflict.evidence_ids) < 2:
                        raise ValueError(
                            "each conflict must cite at least two valid evidence ids"
                        )

            try:
                result = llm.structured(
                    role="strong",
                    system=synthesize_system_prompt(time_sensitive=time_sensitive),
                    user=user,
                    schema=SynthesisDraft,
                    validate=validate_draft,
                )
            except Exception as error:
                errors = list(state.errors)
                errors.append(f"Synthesis failed ({type(error).__name__})")
                reasons = list(state.reasons)
                if Reason.NODE_ERROR not in reasons:
                    reasons.append(Reason.NODE_ERROR)
                run.tracer.event(
                    "note",
                    "synthesize",
                    details={
                        "action": "synthesis_failed",
                        "error_type": type(error).__name__,
                    },
                )
                return {
                    "draft": None,
                    "findings": [],
                    "conflicts": [],
                    "limitations": [],
                    "next_steps": [],
                    "short_answer": None,
                    "errors": errors,
                    "reasons": reasons,
                    "abort": True,
                }

            draft = result.parsed
            findings = [
                Finding(
                    statement=finding.statement,
                    evidence_ids=finding.evidence_ids,
                    support="unchecked",
                )
                for finding in draft.findings
            ]
            return {
                "draft": draft,
                "findings": findings,
                "conflicts": draft.conflicts,
                "limitations": draft.limitations,
                "next_steps": draft.next_steps,
                "short_answer": draft.short_answer,
                "abort": False,
            }

    return synthesize_node
