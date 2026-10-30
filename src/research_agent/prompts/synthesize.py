"""Prompt construction for evidence-only answer synthesis."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping


def synthesize_system_prompt(*, time_sensitive: bool) -> str:
    """Return strong-model instructions constrained to verified evidence."""
    date_instruction = (
        "For time-sensitive questions, mention the dates of the evidence used."
        if time_sensitive
        else ""
    )
    return f"""Answer ONLY from the evidence table in the user message.
Every finding must cite at least one evidence id and state one distinct claim.
Never state a name, number, date, or version unless it appears in the cited
passages. The short_answer must be 2-4 sentences that directly answer the
question, hedged where evidence is thin. State in limitations when a sub-question
has no evidence. Put disagreements between sources in conflicts, with at least
two evidence ids for each conflict. List assumptions and limitations. Provide
2-4 concrete next steps. Never invent evidence ids. {date_instruction}
Return exactly the requested JSON object."""


def synthesize_user_prompt(
    question: str,
    today: str,
    subquestions: Iterable[tuple[str, str]],
    evidence_rows: Iterable[Mapping[str, str | None]],
) -> str:
    """Serialize only the request, plan questions, and allowed evidence fields."""
    questions = [
        {"sub_question_id": sub_question_id, "text": text}
        for sub_question_id, text in subquestions
    ]
    rows = [
        {
            "id": row["id"],
            "sub_question_id": row["sub_question_id"],
            "domain": row["domain"],
            "published_date": row["published_date"],
            "passage": row["passage"],
            "claim": row["claim"],
        }
        for row in evidence_rows
    ]
    return (
        f"Question: {question}\nToday's date: {today}\n"
        f"Plan sub-questions:\n{json.dumps(questions, ensure_ascii=True)}\n"
        f"Evidence table:\n{json.dumps(rows, ensure_ascii=True)}"
    )
