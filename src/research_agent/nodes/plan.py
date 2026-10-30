"""Plan-node factory."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from research_agent.deps import Deps, RunContext
from research_agent.errors import ToolFailure
from research_agent.llm import ResilientLLM
from research_agent.prompts.plan import plan_system_prompt, plan_user_prompt
from research_agent.schemas import AgentState, ErrorClass, Plan, Reason, Strategy

_URL_QUERY_RE = re.compile(r"^https?://", re.IGNORECASE)
_DEPENDENT_REFERENCE_RE = re.compile(
    r"\b(?:these|those|they|them|their|each|the\s+above|above|"
    r"previous(?:ly mentioned)?|aforementioned)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[\w]+(?:[-'][\w]+)*", re.UNICODE)
_NAMED_CANDIDATE_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9]*(?:[.+#-][A-Za-z0-9]+)*"
    r"(?:\s+[A-Z][A-Za-z0-9]*(?:[.+#-][A-Za-z0-9]+)*)*"
)
_TOPIC_STOPWORDS = {
    "a", "an", "and", "are", "about", "best", "compare", "cons", "could",
    "explain", "find", "for", "from", "how", "identify", "in", "is", "list",
    "of", "on", "or", "options", "pros", "research", "show", "the", "top", "two", "using",
    "what", "which", "who", "why", "with",
}
_CANDIDATE_INTRODUCERS = {
    "best", "compare", "candidates", "companies", "identify", "list", "options",
    "the", "top",
}
_WHITESPACE_RE = re.compile(r"\s+")
_TEXT_TRANSLATION = str.maketrans(
    {
        **{chr(codepoint): "-" for codepoint in range(0x2010, 0x2016)},
        "\u2011": "-",
        "\u00a0": " ",
        "\u2007": " ",
        "\u202f": " ",
    }
)


def _normalize_text(text: str) -> str:
    translated = text.translate(_TEXT_TRANSLATION)
    return _WHITESPACE_RE.sub(" ", translated).strip()


def _main_topic_term(question: str) -> str | None:
    """Derive a concise topic token from the end of the user's question."""
    words = _WORD_RE.findall(question.casefold())
    return next(
        (word for word in reversed(words) if word not in _TOPIC_STOPWORDS and not word.isdigit()),
        None,
    )


def _normalized_phrase(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.casefold()))


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = f" {_normalized_phrase(text)} "
    normalized_phrase = f" {_normalized_phrase(phrase)} "
    return normalized_phrase in normalized_text


def _candidate_names(text: str) -> list[str]:
    candidates: list[str] = []
    for match in _NAMED_CANDIDATE_RE.finditer(text):
        candidate = match.group().strip()
        words = candidate.split()
        if words and words[0].casefold() in _CANDIDATE_INTRODUCERS:
            words = words[1:]
            candidate = " ".join(words)
        if candidate and candidate.casefold() not in _CANDIDATE_INTRODUCERS:
            candidates.append(candidate)
    return list(dict.fromkeys(candidates))


def _normalize_plan(plan: Plan) -> None:
    plan.restated_question = _normalize_text(plan.restated_question)
    if plan.clarification_note is not None:
        plan.clarification_note = _normalize_text(plan.clarification_note)
    plan.assumptions = [_normalize_text(item) for item in plan.assumptions]
    for subquestion in plan.sub_questions:
        subquestion.text = _normalize_text(subquestion.text)
        subquestion.queries = [_normalize_text(query) for query in subquestion.queries]
        subquestion.rationale = _normalize_text(subquestion.rationale)


def _validate_plan(plan: Plan) -> None:
    if plan.is_answerable and not plan.sub_questions:
        raise ValueError("an answerable plan must include sub-questions")
    for subquestion in plan.sub_questions:
        if not 1 <= len(subquestion.queries) <= 2:
            raise ValueError("each sub-question must have one or two queries")
        for query in subquestion.queries:
            if _URL_QUERY_RE.match(query):
                raise ValueError("plan queries must not be URLs")


def _plan_warnings(plan: Plan, question: str) -> list[str]:
    warnings: list[str] = []
    topic_term = _main_topic_term(question)
    for subquestion in plan.sub_questions:
        if _DEPENDENT_REFERENCE_RE.search(subquestion.text):
            warnings.append(
                f"{subquestion.id} text uses a dependent reference"
            )
        for query in subquestion.queries:
            if _DEPENDENT_REFERENCE_RE.search(query):
                warnings.append(f"{subquestion.id} query uses a dependent reference")
            words = query.split()
            if not 3 <= len(words) <= 12:
                warnings.append(f"{subquestion.id} query is outside the preferred 3-12 words")
            if topic_term is not None and not _contains_phrase(query, topic_term):
                warnings.append(
                    f"{subquestion.id} query omits the question-derived topic term "
                    f"{topic_term!r}"
                )

    candidate_marker = "candidate list from model knowledge, verified by q1 search"
    if plan.is_answerable and any(
        candidate_marker in assumption.casefold() for assumption in plan.assumptions
    ):
        if not plan.sub_questions or plan.sub_questions[0].id != "q1":
            warnings.append("candidate-list assumption is present without q1 first")
        else:
            candidate_names = _candidate_names(plan.sub_questions[0].text)
            if len(candidate_names) < 2:
                warnings.append("q1 text does not clearly name multiple candidates")
            for subquestion in plan.sub_questions:
                if subquestion.id != "q1" and any(
                    not _contains_phrase(subquestion.text, candidate)
                    for candidate in candidate_names
                ):
                    warnings.append(
                        f"{subquestion.id} text does not repeat all candidate names"
                    )
                if any(
                    not _contains_phrase(query, candidate)
                    for query in subquestion.queries
                    for candidate in candidate_names
                ):
                    warnings.append(
                        f"{subquestion.id} query does not repeat all candidate names"
                    )
    return list(dict.fromkeys(warnings))


def _fallback_query(question: str) -> str:
    words = _WORD_RE.findall(_URL_QUERY_RE.sub(" ", question))
    query = words[:12]
    for fallback_word in ("research", "evidence", "topic"):
        if len(query) >= 3:
            break
        query.append(fallback_word)
    return " ".join(query)


def _fallback_plan(question: str) -> Plan:
    query = _fallback_query(question)
    topic = query or "research evidence topic"
    return Plan.model_validate(
        {
            "restated_question": _normalize_text(question) or "Research question",
            "is_answerable": True,
            "clarification_note": None,
            "assumptions": ["fallback plan used"],
            "sub_questions": [
                {
                    "id": "q1",
                    "text": f"Key aspects of {topic}",
                    "queries": [topic],
                    "strategy": Strategy.search_and_fetch,
                    "time_sensitive": False,
                    "rationale": "Gather evidence about the requested topic",
                },
                {
                    "id": "q2",
                    "text": f"Evidence about {topic}",
                    "queries": [topic],
                    "strategy": Strategy.search_and_fetch,
                    "time_sensitive": False,
                    "rationale": "Find additional evidence about the requested topic",
                },
            ],
        }
    )


def make_plan_node(deps: Deps, run: RunContext) -> Callable[[AgentState], dict[str, Any]]:
    """Build the cheap-model planning node for one run."""
    llm = ResilientLLM(deps.llm, deps.settings, run.budget, run.tracer)

    def plan_node(state: AgentState) -> dict[str, Any]:
        with run.tracer.node_span("plan"):
            def validate_plan(plan: Plan) -> None:
                _normalize_plan(plan)
                _validate_plan(
                    plan,
                )

            try:
                result = llm.structured(
                    role="cheap",
                    system=plan_system_prompt(deps.settings.MAX_SUBQUESTIONS, state.today),
                    user=plan_user_prompt(state.question),
                    schema=Plan,
                    validate=validate_plan,
                )
                plan = result.parsed
            except ToolFailure as error:
                if error.error.error_class is not ErrorClass.SCHEMA:
                    raise
                plan = _fallback_plan(state.question)
                run.tracer.event(
                    "note",
                    "plan",
                    details={
                        "action": "plan_fallback",
                        "validation_error": error.error.message[:300],
                    },
                )
                fallback_errors = [
                    f"Planning step failed: {error.error.message[:240]}"
                ]
                fallback_reasons = [Reason.PLAN_FALLBACK]
            else:
                fallback_errors = []
                fallback_reasons = []

            for warning in _plan_warnings(plan, state.question):
                run.tracer.event(
                    "note",
                    "plan",
                    details={"action": "plan_rule_warning", "warning": warning},
                )
            if len(plan.sub_questions) > deps.settings.MAX_SUBQUESTIONS:
                original_count = len(plan.sub_questions)
                plan = plan.model_copy(
                    update={
                        "sub_questions": plan.sub_questions[
                            : deps.settings.MAX_SUBQUESTIONS
                        ]
                    }
                )
                run.tracer.event(
                    "note",
                    "plan",
                    details={
                        "action": "truncated_subquestions",
                        "before": original_count,
                        "after": len(plan.sub_questions),
                    },
                )
            if not plan.is_answerable:
                return {
                    "plan": plan,
                    "pending_sq_ids": [],
                    "reasons": [*fallback_reasons, Reason.NOT_ANSWERABLE],
                    "errors": fallback_errors,
                    "abort": True,
                }
            return {
                "plan": plan,
                "pending_sq_ids": [item.id for item in plan.sub_questions],
                "reasons": fallback_reasons,
                "errors": fallback_errors,
                "abort": False,
            }

    return plan_node