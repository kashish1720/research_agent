"""Verbatim evidence extraction node and source-text matching helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from research_agent.deps import Deps, RunContext
from research_agent.llm import ResilientLLM
from research_agent.prompts.extract import extract_system_prompt, extract_user_prompt
from research_agent.schemas import (
    AgentState,
    EvidenceItem,
    ExtractionOutput,
    Reason,
)

_ZERO_WIDTH_CHARS = str.maketrans("", "", "\u200b\u200c\u200d\u2060\ufeff")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_PARAGRAPH_RE = re.compile(r"\n[ \t]*\n+")
_SOURCE_ID_RE = re.compile(r"^E(\d+)$")


def normalize_for_match(text: str) -> str:
    """Normalize text for conservative, punctuation-insensitive substring checks."""
    normalized = text.translate(_ZERO_WIDTH_CHARS).casefold()
    normalized = "".join(
        character if character.isalnum() else " "
        for character in normalized
    )
    return " ".join(normalized.split())


def passage_is_verbatim(passage: str, source_text: str) -> bool:
    """Check that the normalized passage occurs contiguously in the full source."""
    normalized_passage = normalize_for_match(passage)
    if not normalized_passage:
        return False
    normalized_source = f" {normalize_for_match(source_text)} "
    return f" {normalized_passage} " in normalized_source


def split_chunks(text: str, max_chunk_chars: int) -> list[str]:
    """Split at paragraph boundaries first, then hard-split oversized paragraphs."""
    if max_chunk_chars < 1:
        raise ValueError("max_chunk_chars must be positive")
    paragraphs = [
        paragraph.strip()
        for paragraph in _PARAGRAPH_RE.split(text)
        if paragraph.strip()
    ]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = [
            paragraph[index : index + max_chunk_chars]
            for index in range(0, len(paragraph), max_chunk_chars)
        ]
        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + 2 + len(piece) <= max_chunk_chars:
                current = f"{current}\n\n{piece}"
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(text)}


def select_chunks(
    chunks: Iterable[str],
    query_text: str,
    max_chars: int,
) -> list[str]:
    """Choose overlapping chunks within a character budget, preserving source order."""
    if max_chars < 0:
        raise ValueError("max_chars must not be negative")
    original_chunks = list(chunks)
    query_tokens = _tokens(query_text)
    ranked = sorted(
        enumerate(original_chunks),
        key=lambda item: (
            -len(query_tokens & _tokens(item[1])),
            item[0],
        ),
    )
    selected: set[int] = set()
    used_chars = 0
    for index, chunk in ranked:
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        if len(chunk) <= remaining:
            selected.add(index)
            used_chars += len(chunk)
        elif not selected:
            selected.add(index)
            used_chars += remaining
    output: list[str] = []
    remaining = max_chars
    for index, chunk in enumerate(original_chunks):
        if index not in selected or remaining <= 0:
            continue
        selected_chunk = chunk[:remaining]
        if selected_chunk:
            output.append(selected_chunk)
            remaining -= len(selected_chunk)
    return output


def _next_evidence_number(evidence: list[EvidenceItem]) -> int:
    return max(
        (
            int(match.group(1))
            for item in evidence
            if (match := _SOURCE_ID_RE.match(item.id)) is not None
        ),
        default=len(evidence),
    ) + 1


def _source_order(source_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"S(\d+)", source_id)
    return (int(match.group(1)), source_id) if match else (2**31, source_id)


def _query_text(subquestion_text: str, queries: Iterable[str]) -> str:
    return " ".join((subquestion_text, *queries))


def _log_rejection(
    run: RunContext,
    *,
    source_id: str,
    reason: str,
    passage: str,
) -> None:
    run.tracer.event(
        "note",
        "extract",
        details={
            "action": "rejected_passage",
            "source_id": source_id,
            "reason": reason,
            "passage_prefix": passage[:80],
        },
    )


def make_extract_node(
    deps: Deps,
    run: RunContext,
) -> Callable[[AgentState], dict[str, Any]]:
    """Build the cheap-model extraction node for one run."""
    llm = ResilientLLM(deps.llm, deps.settings, run.budget, run.tracer)

    def extract_node(state: AgentState) -> dict[str, Any]:
        with run.tracer.node_span("extract"):
            if state.plan is None:
                raise ValueError("extraction requires a research plan")
            evidence = list(state.evidence)
            reasons = list(state.reasons)
            errors = list(state.errors)
            stats = dict(state.stats)
            next_evidence_number = _next_evidence_number(evidence)
            accepted_count = 0
            snippet_only_count = 0

            for subquestion_id in state.pending_sq_ids:
                subquestion = next(
                    (
                        item
                        for item in state.plan.sub_questions
                        if item.id == subquestion_id
                    ),
                    None,
                )
                if subquestion is None:
                    errors.append(f"Extraction skipped unknown sub-question {subquestion_id}")
                    continue
                source_list = [
                    source
                    for source in sorted(
                        state.sources.values(),
                        key=lambda candidate: _source_order(candidate.id),
                    )
                    if subquestion_id in source.sub_question_ids
                ]
                source_by_id = {source.id: source for source in source_list}
                query_text = _query_text(subquestion.text, subquestion.queries)
                prompt_sources: list[tuple[str, str, str]] = []
                source_texts: dict[str, str] = {}
                for source in source_list:
                    full_text = source.text or ""
                    if not full_text:
                        continue
                    source_texts[source.id] = full_text
                    chunks = (
                        [full_text]
                        if source.snippet_only
                        else split_chunks(full_text, deps.settings.EXTRACT_CHUNK_CHARS)
                    )
                    selected = select_chunks(
                        chunks,
                        query_text,
                        deps.settings.EXTRACT_MAX_CHARS_PER_SOURCE,
                    )
                    if selected:
                        prompt_sources.append(
                            (source.id, source.url, "\n\n".join(selected))
                        )
                provided_source_ids = frozenset(
                    source_tuple[0] for source_tuple in prompt_sources
                )

                def validate_output(
                    output: ExtractionOutput,
                    valid_source_ids: frozenset[str] = provided_source_ids,
                ) -> None:
                    for item in output.items:
                        if item.source_id not in valid_source_ids:
                            raise ValueError(
                                f"unknown source_id in extraction output: {item.source_id}"
                            )

                try:
                    result = llm.structured(
                        role="cheap",
                        system=extract_system_prompt(
                            max_evidence_per_source=deps.settings.MAX_EVIDENCE_PER_SOURCE,
                            passage_min_words=deps.settings.PASSAGE_MIN_WORDS,
                            passage_max_words=deps.settings.PASSAGE_MAX_WORDS,
                        ),
                        user=extract_user_prompt(subquestion.text, prompt_sources),
                        schema=ExtractionOutput,
                        validate=validate_output,
                    )
                except Exception as error:
                    errors.append(
                        f"Extraction failed for {subquestion_id} "
                        f"({type(error).__name__})"
                    )
                    if Reason.EXTRACTION_FAILED not in reasons:
                        reasons.append(Reason.EXTRACTION_FAILED)
                    run.tracer.event(
                        "note",
                        "extract",
                        details={
                            "action": "subquestion_extraction_failed",
                            "sub_question_id": subquestion_id,
                            "error_type": type(error).__name__,
                        },
                    )
                    continue

                counts_by_source: dict[str, int] = {}
                for existing_item in evidence:
                    if existing_item.sub_question_id == subquestion_id:
                        counts_by_source[existing_item.source_id] = (
                            counts_by_source.get(existing_item.source_id, 0) + 1
                        )
                duplicate_keys = {
                    (item.sub_question_id, normalize_for_match(item.passage))
                    for item in evidence
                }
                for item in result.parsed.items:
                    source = source_by_id.get(item.source_id)
                    source_text = source_texts.get(item.source_id)
                    if source is None or source_text is None:
                        continue
                    word_count = len(item.passage.split())
                    if not (
                        deps.settings.PASSAGE_MIN_WORDS
                        <= word_count
                        <= deps.settings.PASSAGE_MAX_WORDS
                    ):
                        stats["rejected_length"] = stats.get("rejected_length", 0) + 1
                        _log_rejection(
                            run,
                            source_id=item.source_id,
                            reason="length",
                            passage=item.passage,
                        )
                        continue
                    if (
                        "..." in item.passage
                        or "…" in item.passage
                        or not passage_is_verbatim(item.passage, source_text)
                    ):
                        stats["rejected_not_verbatim"] = (
                            stats.get("rejected_not_verbatim", 0) + 1
                        )
                        _log_rejection(
                            run,
                            source_id=item.source_id,
                            reason="not_verbatim",
                            passage=item.passage,
                        )
                        continue
                    duplicate_key = (subquestion_id, normalize_for_match(item.passage))
                    if duplicate_key in duplicate_keys:
                        stats["rejected_duplicate"] = (
                            stats.get("rejected_duplicate", 0) + 1
                        )
                        _log_rejection(
                            run,
                            source_id=item.source_id,
                            reason="duplicate",
                            passage=item.passage,
                        )
                        continue
                    if counts_by_source.get(item.source_id, 0) >= (
                        deps.settings.MAX_EVIDENCE_PER_SOURCE
                    ):
                        stats["rejected_per_source_cap"] = (
                            stats.get("rejected_per_source_cap", 0) + 1
                        )
                        _log_rejection(
                            run,
                            source_id=item.source_id,
                            reason="per_source_cap",
                            passage=item.passage,
                        )
                        continue

                    evidence.append(
                        EvidenceItem(
                            id=f"E{next_evidence_number}",
                            sub_question_id=subquestion_id,
                            source_id=source.id,
                            domain=source.domain,
                            published_date=source.published_date,
                            passage=item.passage,
                            claim=item.claim,
                            snippet_only=source.snippet_only,
                        )
                    )
                    next_evidence_number += 1
                    accepted_count += 1
                    if source.snippet_only:
                        snippet_only_count += 1
                    counts_by_source[item.source_id] = (
                        counts_by_source.get(item.source_id, 0) + 1
                    )
                    duplicate_keys.add(duplicate_key)

            if accepted_count and snippet_only_count / accepted_count > 0.5:
                if Reason.SNIPPET_ONLY_EVIDENCE not in reasons:
                    reasons.append(Reason.SNIPPET_ONLY_EVIDENCE)
            return {
                "evidence": evidence,
                "pending_sq_ids": [],
                "reasons": reasons,
                "errors": errors,
                "stats": stats,
            }

    return extract_node
