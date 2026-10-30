"""Coverage calculation, sufficiency routing, and bounded query refinement."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from research_agent.budget import Budget
from research_agent.config import Settings
from research_agent.deps import Deps, RunContext
from research_agent.filters import registered_domain
from research_agent.llm import ResilientLLM
from research_agent.prompts.refine import refine_system_prompt, refine_user_prompt
from research_agent.schemas import (
    AgentState,
    Coverage,
    EvidenceItem,
    Plan,
    Reason,
    RefinedQueries,
    SubQuestionCoverage,
)

_URL_RE = re.compile(r"^(?:[a-z][a-z0-9+.-]*://|www\.)", re.IGNORECASE)
_QUERY_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def compute_coverage(plan: Plan, evidence: list[EvidenceItem]) -> Coverage:
    """Count evidence and distinct registered domains for each planned question."""
    per_subquestion: list[SubQuestionCoverage] = []
    for subquestion in plan.sub_questions:
        matching = [
            item for item in evidence if item.sub_question_id == subquestion.id
        ]
        domains = {registered_domain(item.domain) for item in matching if item.domain}
        per_subquestion.append(
            SubQuestionCoverage(
                id=subquestion.id,
                text=subquestion.text,
                evidence_count=len(matching),
                independent_domains=len(domains),
            )
        )
    return Coverage(
        total=len(per_subquestion),
        covered=sum(item.evidence_count > 0 for item in per_subquestion),
        per_subquestion=per_subquestion,
    )


def decide(
    coverage: Coverage,
    *,
    refine_loops: int,
    settings: Settings,
    budget: Budget,
    allow_refine: bool = True,
) -> str:
    """Apply the SPEC coverage thresholds and available budget/time exactly."""
    coverage_ratio = coverage.covered / coverage.total if coverage.total else 0.0
    if coverage_ratio >= settings.TARGET_COVERAGE:
        return "synthesize"
    if (
        allow_refine
        and refine_loops < settings.MAX_REFINE_LOOPS
        and budget.can_search()
        and budget.can_llm()
        and budget.time_left() > 0
    ):
        return "refine"
    if coverage_ratio >= settings.MIN_COVERAGE_TO_ANSWER:
        return "synthesize"
    return "insufficient"


def _normalize_query(query: str) -> str:
    return " ".join(_QUERY_TOKEN_RE.findall(query.casefold()))


def _decision_reasons(
    decision: str,
    coverage: Coverage,
    reasons: list[Reason],
) -> list[Reason]:
    if decision == "insufficient":
        if Reason.INSUFFICIENT_EVIDENCE not in reasons:
            reasons.append(Reason.INSUFFICIENT_EVIDENCE)
    elif decision == "synthesize" and coverage.covered < coverage.total:
        if Reason.UNCOVERED_SUBQUESTIONS not in reasons:
            reasons.append(Reason.UNCOVERED_SUBQUESTIONS)
    return reasons


def make_sufficiency_node(
    deps: Deps,
    run: RunContext,
) -> Callable[[AgentState], dict[str, Any]]:
    """Build the node that routes on evidence coverage and optionally refines."""
    llm = ResilientLLM(deps.llm, deps.settings, run.budget, run.tracer)

    def sufficiency_node(state: AgentState) -> dict[str, Any]:
        with run.tracer.node_span("sufficiency"):
            if state.plan is None:
                raise ValueError("sufficiency requires a research plan")
            coverage = compute_coverage(state.plan, state.evidence)
            decision = decide(
                coverage,
                refine_loops=state.refine_loops,
                settings=deps.settings,
                budget=run.budget,
            )
            reasons = list(state.reasons)
            errors = list(state.errors)

            if decision != "refine":
                reasons = _decision_reasons(decision, coverage, reasons)
                return {
                    "route": decision,
                    "reasons": reasons,
                }

            covered_ids = {
                item.id for item in coverage.per_subquestion if item.evidence_count
            }
            uncovered = [
                item for item in state.plan.sub_questions if item.id not in covered_ids
            ]
            uncovered_ids = {item.id for item in uncovered}
            old_queries = {
                item.id: {_normalize_query(query) for query in item.queries}
                for item in uncovered
            }

            def validate_refinement(output: RefinedQueries) -> None:
                returned_ids = [item.sub_question_id for item in output.items]
                if any(item_id not in uncovered_ids for item_id in returned_ids):
                    raise ValueError("refinement returned an unknown sub-question id")
                if set(returned_ids) != uncovered_ids or len(returned_ids) != len(
                    uncovered_ids
                ):
                    raise ValueError(
                        "refinement must return exactly one item for every uncovered sub-question"
                    )
                for item in output.items:
                    prior = old_queries[item.sub_question_id]
                    seen = set(prior)
                    for query in item.queries:
                        if not query.strip():
                            raise ValueError("refined queries must not be blank")
                        if _URL_RE.match(query.strip()):
                            raise ValueError("refined queries must not be URLs")
                        normalized_query = _normalize_query(query)
                        if normalized_query in seen:
                            raise ValueError(
                                f"refined query repeats a previous query for {item.sub_question_id}"
                            )
                        seen.add(normalized_query)

            user = refine_user_prompt(
                (
                    (item.id, item.text, list(item.queries))
                    for item in uncovered
                ),
                "Prior searches did not provide evidence for these questions.",
            )
            try:
                result = llm.structured(
                    role="cheap",
                    system=refine_system_prompt(),
                    user=user,
                    schema=RefinedQueries,
                    validate=validate_refinement,
                )
            except Exception as error:
                errors.append(f"Query refinement failed ({type(error).__name__})")
                run.tracer.event(
                    "note",
                    "sufficiency",
                    details={
                        "action": "refinement_failed",
                        "error_type": type(error).__name__,
                    },
                )
                decision = decide(
                    coverage,
                    refine_loops=state.refine_loops,
                    settings=deps.settings,
                    budget=run.budget,
                    allow_refine=False,
                )
                reasons = _decision_reasons(decision, coverage, reasons)
                return {
                    "route": decision,
                    "pending_sq_ids": [],
                    "refine_loops": state.refine_loops,
                    "reasons": reasons,
                    "errors": errors,
                }

            refined_queries = {
                item.sub_question_id: item.queries for item in result.parsed.items
            }
            updated_plan = state.plan.model_copy(
                update={
                    "sub_questions": [
                        question.model_copy(
                            update={"queries": refined_queries[question.id]}
                        )
                        if question.id in refined_queries
                        else question
                        for question in state.plan.sub_questions
                    ]
                }
            )
            return {
                "route": "refine",
                "plan": updated_plan,
                "pending_sq_ids": [item.id for item in uncovered],
                "refine_loops": state.refine_loops + 1,
                "reasons": reasons,
                "errors": errors,
            }

    return sufficiency_node
