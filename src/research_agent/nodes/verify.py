"""Deterministic and entailment verification for synthesized findings."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any

from research_agent.deps import Deps, RunContext
from research_agent.errors import BudgetExceeded, ToolFailure
from research_agent.llm import ResilientLLM
from research_agent.prompts.verify import verify_system_prompt, verify_user_prompt
from research_agent.schemas import (
    AgentState,
    Conflict,
    EntailmentOutput,
    Finding,
    Reason,
)

_NUMBER_RE = re.compile(
    r"(?<![\w.])(?P<number>\d[\d,]*(?:\.\d+)?)(?P<percent>\s*(?:%|percent\b))?",
    re.IGNORECASE,
)


def _normalize_evidence_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.replace("\u00a0", " ")
    return normalized


def _number_key(number: str, percent: str | None) -> tuple[str, bool]:
    compact = number.replace(",", "")
    try:
        normalized_number = format(Decimal(compact).normalize(), "f")
    except InvalidOperation:
        normalized_number = compact
    return normalized_number, bool(percent and percent.strip())


def unmatched_numeric_tokens(statement: str, passages: list[str]) -> list[str]:
    """Return statement numbers that do not occur in its cited passages."""
    statement_matches = [
        (match.group(0), _number_key(match.group("number"), match.group("percent")))
        for match in _NUMBER_RE.finditer(statement)
    ]
    cited_text = _normalize_evidence_text(" ".join(passages))
    cited_keys = {
        _number_key(match.group("number"), match.group("percent"))
        for match in _NUMBER_RE.finditer(cited_text)
    }
    return [raw for raw, key in statement_matches if key not in cited_keys]


def _add_reason(reasons: list[Reason], reason: Reason) -> None:
    if reason not in reasons:
        reasons.append(reason)


def make_verify_node(
    deps: Deps,
    run: RunContext,
) -> Callable[[AgentState], dict[str, Any]]:
    """Build the one-call, batched verifier with deterministic safeguards."""
    llm = ResilientLLM(deps.llm, deps.settings, run.budget, run.tracer)

    def verify_node(state: AgentState) -> dict[str, Any]:
        with run.tracer.node_span("verify"):
            evidence_by_id = {item.id: item for item in state.evidence}
            reasons = list(state.reasons)
            limitations = list(state.limitations)
            original_findings = list(state.findings)
            dropped = False

            findings: list[Finding] = []
            for finding in original_findings:
                if not finding.evidence_ids or any(
                    evidence_id not in evidence_by_id for evidence_id in finding.evidence_ids
                ):
                    dropped = True
                    run.tracer.event(
                        "note",
                        "verify",
                        details={"action": "finding_dropped", "reason": "invalid_evidence_ids"},
                    )
                    continue
                if deps.settings.STRICT_NUMERIC_CHECK:
                    passages = [
                        evidence_by_id[evidence_id].passage
                        for evidence_id in finding.evidence_ids
                    ]
                    unmatched = unmatched_numeric_tokens(finding.statement, passages)
                    if unmatched:
                        dropped = True
                        run.tracer.event(
                            "note",
                            "verify",
                            details={
                                "action": "finding_dropped",
                                "reason": "numeric_mismatch",
                                "unmatched_numbers": unmatched,
                            },
                        )
                        continue
                findings.append(finding.model_copy(update={"support": "unchecked"}))

            valid_evidence_ids = set(evidence_by_id)
            conflicts: list[Conflict] = []
            for conflict in state.conflicts:
                cited = [
                    evidence_id
                    for evidence_id in conflict.evidence_ids
                    if evidence_id in valid_evidence_ids
                ]
                if len(cited) >= 2:
                    conflicts.append(
                        conflict.model_copy(update={"evidence_ids": cited})
                    )

            if dropped:
                _add_reason(reasons, Reason.FINDINGS_DROPPED)

            verification_complete = True
            if findings:
                prompt_findings = [
                    {
                        "index": index,
                        "statement": finding.statement,
                        "passages": [
                            evidence_by_id[evidence_id].passage
                            for evidence_id in finding.evidence_ids
                        ],
                    }
                    for index, finding in enumerate(findings)
                ]
                expected_indices = set(range(len(findings)))

                def validate_output(output: EntailmentOutput) -> None:
                    indices = [item.index for item in output.items]
                    if set(indices) != expected_indices or len(indices) != len(expected_indices):
                        raise ValueError(
                            "entailment output must include every finding index exactly once"
                        )

                try:
                    run.budget.check_time()
                    result = llm.structured(
                        role="cheap",
                        system=verify_system_prompt(),
                        user=verify_user_prompt(prompt_findings),
                        schema=EntailmentOutput,
                        validate=validate_output,
                    )
                except (ToolFailure, BudgetExceeded) as error:
                    verification_complete = False
                    _add_reason(reasons, Reason.VERIFICATION_UNAVAILABLE)
                    sentence = (
                        "Entailment verification was unavailable; surviving findings "
                        "remain unchecked."
                    )
                    if sentence not in limitations:
                        limitations.append(sentence)
                    run.tracer.event(
                        "note",
                        "verify",
                        details={
                            "action": "verification_unavailable",
                            "error_type": type(error).__name__,
                        },
                    )
                else:
                    verdicts = {item.index: item for item in result.parsed.items}
                    verified: list[Finding] = []
                    for index, finding in enumerate(findings):
                        item = verdicts[index]
                        if item.verdict == "unsupported":
                            dropped = True
                            run.tracer.event(
                                "note",
                                "verify",
                                details={
                                    "action": "finding_dropped",
                                    "reason": "unsupported",
                                    "index": index,
                                },
                            )
                        elif item.verdict == "partial":
                            verified.append(
                                finding.model_copy(update={"support": "partial"})
                            )
                            sentence = (
                                f"Finding {index + 1} is only partially supported: "
                                f"{item.reason}"
                            )
                            if sentence not in limitations:
                                limitations.append(sentence)
                        else:
                            verified.append(
                                finding.model_copy(update={"support": "supported"})
                            )
                    findings = verified

            if dropped:
                _add_reason(reasons, Reason.FINDINGS_DROPPED)
            if original_findings and not findings:
                _add_reason(reasons, Reason.ALL_FINDINGS_UNSUPPORTED)

            short_answer = state.short_answer or ""
            finding_numbers = {
                _number_key(match.group("number"), match.group("percent"))
                for finding in findings
                for match in _NUMBER_RE.finditer(finding.statement)
            }
            unmatched_answer_numbers = [
                match.group(0)
                for match in _NUMBER_RE.finditer(short_answer)
                if _number_key(match.group("number"), match.group("percent"))
                not in finding_numbers
            ]
            if unmatched_answer_numbers:
                if findings:
                    short_answer = (
                        "Based on the verified findings below: "
                        + " ".join(item.statement for item in findings[:2])
                    )
                else:
                    short_answer = "Reliable information was not found in the verified findings."
                limitation = (
                    "The short answer included numeric details not present in the "
                    "surviving findings, so it was replaced."
                )
                if limitation not in limitations:
                    limitations.append(limitation)

            return {
                "findings": findings,
                "conflicts": conflicts,
                "short_answer": short_answer,
                "limitations": limitations,
                "reasons": reasons,
                "abort": bool(original_findings and not findings),
                "stats": {
                    **state.stats,
                    "verification_complete": int(verification_complete),
                },
            }

    return verify_node
