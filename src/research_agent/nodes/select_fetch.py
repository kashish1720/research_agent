"""Sequential source selection and guarded page fetching."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict

from research_agent.deps import Deps, RunContext
from research_agent.errors import BudgetExceeded, ToolFailure
from research_agent.filters import (
    dedupe_results,
    normalize_url,
    registered_domain,
    scan_for_injection,
    score_result,
)
from research_agent.guard import Memo, guarded_call
from research_agent.schemas import (
    AgentState,
    FetchStatus,
    Reason,
    SearchResult,
    Source,
    Strategy,
)

_SOURCE_ID_RE = re.compile(r"^S(\d+)$")


class _FetchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str


@dataclass
class _Candidate:
    result: SearchResult
    score: float
    flags: list[str]
    sub_question_ids: list[str] = field(default_factory=list)
    should_fetch: bool = False


def _next_source_number(sources: dict[str, Source]) -> int:
    highest_id = max(
        (
            int(match.group(1))
            for source in sources.values()
            if (match := _SOURCE_ID_RE.match(source.id)) is not None
        ),
        default=0,
    )
    return max(highest_id, len(sources)) + 1


def _add_subquestion(candidate: _Candidate, sub_question_id: str) -> None:
    if sub_question_id not in candidate.sub_question_ids:
        candidate.sub_question_ids.append(sub_question_id)


def make_select_fetch_node(
    deps: Deps,
    run: RunContext,
) -> Callable[[AgentState], dict[str, Any]]:
    """Build the node that scores, selects, fetches, and sanitizes sources."""
    memo = Memo()

    def select_fetch_node(state: AgentState) -> dict[str, Any]:
        with run.tracer.node_span("select_fetch"):
            if state.plan is None:
                raise ValueError("source selection requires a research plan")
            try:
                today = date.fromisoformat(state.today)
            except ValueError as error:
                raise ValueError("state.today must be an ISO date") from error

            sources = dict(state.sources)
            reasons = list(state.reasons)
            existing_urls = {normalize_url(source.url) for source in sources.values()}
            selected_by_url: dict[str, _Candidate] = {}
            subquestion_order = {
                item.id: index for index, item in enumerate(state.plan.sub_questions)
            }

            for sub_question_id in state.pending_sq_ids:
                subquestion = next(
                    (
                        item
                        for item in state.plan.sub_questions
                        if item.id == sub_question_id
                    ),
                    None,
                )
                if subquestion is None:
                    run.tracer.event(
                        "note",
                        "select_fetch",
                        details={
                            "action": "unknown_sub_question",
                            "id": sub_question_id,
                        },
                    )
                    continue
                scored: dict[str, tuple[SearchResult, float, list[str]]] = {}
                for result in state.search_results.get(sub_question_id, []):
                    normalized_url = normalize_url(result.url)
                    if normalized_url in existing_urls:
                        continue
                    score, flags = score_result(
                        result,
                        sub_question_text=subquestion.text,
                        queries=subquestion.queries,
                        time_sensitive=subquestion.time_sensitive,
                        today=today,
                        settings=deps.settings,
                    )
                    previous = scored.get(normalized_url)
                    if previous is None or score > previous[1]:
                        scored[normalized_url] = result, score, flags
                unique_results = dedupe_results(
                    (entry[0] for entry in scored.values()),
                    score_fn=lambda candidate, score_map=scored: score_map[
                        normalize_url(candidate.url)
                    ][1],
                )
                ranked = sorted(
                    (
                        (
                            result,
                            *scored[normalize_url(result.url)][1:],
                        )
                        for result in unique_results
                    ),
                    key=lambda item: (-item[1], normalize_url(item[0].url)),
                )
                ranked = [
                    item
                    for item in ranked
                    if item[1] >= deps.settings.MIN_SOURCE_SCORE
                ]
                if subquestion.strategy is Strategy.search_and_fetch:
                    top = ranked[: deps.settings.TOP_K_FETCH_PER_SUBQ]
                    should_fetch = True
                else:
                    top = ranked[: deps.settings.SNIPPET_ONLY_TOP_K]
                    should_fetch = False

                for result, score, flags in top:
                    normalized_url = normalize_url(result.url)
                    candidate = selected_by_url.get(normalized_url)
                    if candidate is None:
                        candidate = _Candidate(
                            result=result,
                            score=score,
                            flags=list(flags),
                        )
                        selected_by_url[normalized_url] = candidate
                    elif score > candidate.score:
                        candidate.result = result
                        candidate.score = score
                        candidate.flags = list(flags)
                    _add_subquestion(candidate, sub_question_id)
                    candidate.should_fetch = candidate.should_fetch or should_fetch

            if selected_by_url:
                scored_by_url = {
                    url: candidate.score for url, candidate in selected_by_url.items()
                }
                unique_candidates = dedupe_results(
                    (candidate.result for candidate in selected_by_url.values()),
                    score_fn=lambda result: scored_by_url[normalize_url(result.url)],
                )
                retained_urls = {normalize_url(result.url) for result in unique_candidates}
                selected_by_url = {
                    url: candidate
                    for url, candidate in selected_by_url.items()
                    if url in retained_urls
                }

            ordered_candidates = sorted(
                selected_by_url.items(),
                key=lambda item: (
                    min(subquestion_order.get(sq_id, len(subquestion_order))
                        for sq_id in item[1].sub_question_ids),
                    -item[1].score,
                    item[0],
                ),
            )
            next_source_number = _next_source_number(sources)
            fetch_budget_exhausted = False

            for normalized_url, candidate in ordered_candidates:
                run.allowlist.add([normalized_url])
                title = candidate.result.title
                text = candidate.result.snippet
                published_date = candidate.result.published_date
                fetched = False
                snippet_only = True
                flags = list(candidate.flags)
                fetch_status: str | None = None

                if candidate.should_fetch and not fetch_budget_exhausted:
                    try:
                        fetch_result = guarded_call(
                            tool="fetch",
                            fn=lambda args: run.fetch_fn(args.url),
                            args_model=_FetchArgs,
                            raw_args={"url": normalized_url},
                            budget=run.budget,
                            tracer=run.tracer,
                            budget_kind="fetch",
                            memo=memo,
                        )
                    except BudgetExceeded as exhausted:
                        fetch_budget_exhausted = True
                        if Reason.BUDGET_EXHAUSTED not in reasons:
                            reasons.append(Reason.BUDGET_EXHAUSTED)
                        run.tracer.event(
                            "note",
                            "select_fetch",
                            details={
                                "action": "budget_exhausted",
                                "resource": exhausted.resource,
                            },
                        )
                    except ToolFailure as failure:
                        fetch_status = (
                            str(failure.error.http_status)
                            if failure.error.http_status is not None
                            else failure.error.error_class.value
                        )
                    else:
                        if fetch_result.status is FetchStatus.OK:
                            text = fetch_result.text
                            title = fetch_result.title or title
                            published_date = fetch_result.published_date or published_date
                            fetched = True
                            snippet_only = False
                        else:
                            fetch_status = fetch_result.status.value

                if not fetched:
                    if fetch_status is not None:
                        flags.append(f"fetch_failed:{fetch_status}")
                    if len(text) < deps.settings.MIN_SNIPPET_CHARS:
                        continue

                cleaned_text, matched_names = scan_for_injection(text)
                if matched_names:
                    flags.append("possible_injection")
                    text = cleaned_text
                    if len(matched_names) >= deps.settings.INJECTION_DROP_THRESHOLD:
                        run.tracer.event(
                            "note",
                            "select_fetch",
                            details={
                                "action": "source_dropped_injection",
                                "url": normalized_url,
                                "matches": len(matched_names),
                            },
                        )
                        continue
                else:
                    text = cleaned_text

                source_id = f"S{next_source_number}"
                next_source_number += 1
                source = Source(
                    id=source_id,
                    url=normalized_url,
                    domain=registered_domain(normalized_url),
                    title=title,
                    published_date=published_date,
                    quality_score=min(1.0, max(0.0, candidate.score)),
                    fetched=fetched,
                    snippet_only=snippet_only,
                    flags=list(dict.fromkeys(flags)),
                    sub_question_ids=candidate.sub_question_ids,
                    text=text,
                )
                sources[source_id] = source

            return {"sources": sources, "reasons": reasons}

    return select_fetch_node
