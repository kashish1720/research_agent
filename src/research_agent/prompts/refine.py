"""Prompt construction for refining uncovered research queries."""

from __future__ import annotations

from collections.abc import Iterable


def refine_system_prompt() -> str:
    """Return tool-less instructions for producing distinct search queries."""
    return """You refine research queries for uncovered sub-questions.
Use only the supplied sub-question text and previous queries. Return one or two
new, non-URL queries for each supplied sub-question. Every query must be distinct
from all previous queries for that sub-question. Return the requested JSON object
only."""


def refine_user_prompt(
    subquestions: Iterable[tuple[str, str, list[str]]],
    note: str,
) -> str:
    """Include only uncovered sub-question text, its previous queries, and a note."""
    entries = [
        {
            "sub_question_id": sub_question_id,
            "text": text,
            "previous_queries": previous_queries,
        }
        for sub_question_id, text, previous_queries in subquestions
    ]
    return f"Refinement note: {note}\nUncovered sub-questions:\n{entries!r}"
