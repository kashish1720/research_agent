"""Human-readable rich output with a plain-text fallback."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import TextIO

from research_agent.schemas import FinalAnswer


def _items(values: Sequence[str]) -> str:
    return "\n".join(f"- {value}" for value in values) if values else "- None"


def render_answer(answer: FinalAnswer, *, output: TextIO | None = None) -> None:
    """Render a final answer in the CLI's stable section order."""
    stream = output or sys.stdout
    sections = [
        ("Question", answer.user_question),
        ("Short answer", answer.short_answer),
        (
            "Key findings",
            "\n".join(
                f"- [{finding.support}] {finding.statement} "
                f"[{', '.join(finding.evidence_ids)}]"
                for finding in answer.key_findings
            )
            or "- None",
        ),
        (
            "Sources used",
            "\n".join(
                "- "
                f"{source.id} | {source.domain} | {source.title} | {source.url} | "
                f"{source.published_date or 'date unknown'} | "
                f"flags: {', '.join(source.flags) or 'none'}"
                for source in answer.sources_used
            )
            or "- None",
        ),
        (
            "Confidence",
            "\n".join(
                [
                    f"Level: {answer.confidence.level}",
                    f"Coverage: {answer.confidence.coverage_ratio:.0%}",
                    f"Average source quality: {answer.confidence.avg_source_quality:.2f}",
                    f"Minimum independent domains: "
                    f"{answer.confidence.min_independent_domains}",
                    f"Unresolved conflicts: {answer.confidence.unresolved_conflicts}",
                    f"Snippet-only ratio: {answer.confidence.snippet_only_ratio:.0%}",
                    f"Verification complete: {answer.confidence.verification_complete}",
                    "Reasons:",
                    _items(answer.confidence.reasons),
                ]
            ),
        ),
        (
            "Conflicts",
            "\n".join(
                f"- {conflict.topic}: {conflict.description} "
                f"[{', '.join(conflict.evidence_ids)}]"
                for conflict in answer.conflicts
            )
            or "- None",
        ),
        (
            "Limitations and assumptions",
            _items(answer.limitations_and_assumptions),
        ),
        ("Suggested next steps", _items(answer.suggested_next_steps)),
        (
            "Run info",
            "\n".join(
                [
                    f"Status: {answer.status}",
                    f"Reasons: {', '.join(str(reason) for reason in answer.reasons) or 'none'}",
                    f"Usage: {answer.usage.llm_calls} LLM calls, "
                    f"{answer.usage.input_tokens} input / "
                    f"{answer.usage.output_tokens} output tokens, "
                    f"{answer.usage.searches} searches, {answer.usage.fetches} fetches, "
                    f"{answer.usage.tool_retries} retries, {answer.usage.latency_s:.2f}s",
                    "Estimated cost: "
                    + (
                        f"${answer.usage.est_cost_usd:.6f}"
                        if answer.usage.est_cost_usd is not None
                        else "not configured"
                    ),
                    f"Run id: {answer.run_id}",
                ]
            ),
        ),
    ]

    try:
        from rich.console import Console
        from rich.panel import Panel
    except ImportError:
        stream.write(
            "\n\n".join(f"{title}\n{body}" for title, body in sections) + "\n"
        )
        return

    console = Console(
        file=stream,
        force_terminal=False,
        no_color=True,
        width=100,
        markup=False,
        highlight=False,
    )
    for title, body in sections:
        console.print(Panel(body, title=title, border_style="blue"))
