"""Tests for the offline manual review sheet generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.review_sheet import build_review_sheet, candidate_names, main


def test_review_sheet_generation_keeps_human_verdict_fields_empty() -> None:
    sample_path = Path(__file__).parent / "fixtures" / "review_sheet_sample.json"

    sheet, summary, _candidate_rows = build_review_sheet([sample_path])

    assert "| review_sheet_sample | 1 | 1/1 findings resolve; broken-citation rate 0% |" in summary
    assert "pipeline self-check, not independent" in summary
    assert "0/1 (0%)" in summary
    finding_row = next(line for line in sheet.splitlines() if "| Acme and Qdrant" in line)
    assert finding_row.endswith("|  |  |")
    assert "https://docs.example.com/guide" in finding_row


def test_missing_metrics_report_not_available(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sample_path = tmp_path / "missing.json"
    sample_path.write_text(
        json.dumps(
            {
                "key_findings": [
                    {"statement": "Claim without saved evidence", "evidence_ids": ["E1"]}
                ]
            }
        ),
        encoding="utf-8",
    )

    result = main([str(sample_path), "--out", str(tmp_path / "review.md")])
    captured = capsys.readouterr()

    assert result == 0
    assert "not available: no saved evidence ledger or resolvable source citations" in captured.out
    assert "not available: no sources_used/sources field" in captured.out


def test_candidate_names_are_deduplicated_case_insensitively() -> None:
    findings = [
        {"statement": "Qdrant helps Acme teams."},
        {"statement": "Acme also uses PostgreSQL."},
    ]

    assert candidate_names(findings) == ["Qdrant", "Acme", "PostgreSQL"]


def test_candidates_are_listed_in_summary_and_sheet() -> None:
    sample_path = Path(__file__).parent / "fixtures" / "review_sheet_sample.json"

    sheet, summary, candidates = build_review_sheet([sample_path], include_candidates=True)

    assert candidates == [("review_sheet_sample", ["Acme", "Qdrant"])]
    assert "Candidates review_sheet_sample: Acme, Qdrant" in summary
    assert "## Candidate-name review aids" in sheet
    assert "- **review_sheet_sample:** Acme, Qdrant" in sheet
