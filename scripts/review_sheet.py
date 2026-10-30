"""Generate a human review sheet and conservative offline sample checks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_CANDIDATE_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[.+#-][A-Za-z0-9]+)*\b")
_SUPPORT_MARKERS = ("supported", "partial", "unchecked")


def _markdown(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\r", " ").replace("\n", " ")


def _evidence_items(sample: dict[str, Any]) -> tuple[bool, dict[str, dict[str, Any]]]:
    if "evidence" not in sample:
        return False, {}
    raw_evidence = sample["evidence"]
    if isinstance(raw_evidence, dict):
        return True, {
            str(evidence_id): item
            for evidence_id, item in raw_evidence.items()
            if isinstance(item, dict)
        }
    if isinstance(raw_evidence, list):
        return True, {
            str(item["id"]): item
            for item in raw_evidence
            if isinstance(item, dict) and item.get("id") is not None
        }
    return True, {}


def _sources(sample: dict[str, Any]) -> tuple[bool, dict[str, dict[str, Any]]]:
    if "sources_used" in sample:
        raw_sources = sample["sources_used"]
    elif "sources" in sample:
        raw_sources = sample["sources"]
    else:
        return False, {}
    if isinstance(raw_sources, dict):
        return True, {
            str(source_id): source
            for source_id, source in raw_sources.items()
            if isinstance(source, dict)
        }
    if isinstance(raw_sources, list):
        return True, {
            str(source["id"]): source
            for source in raw_sources
            if isinstance(source, dict) and source.get("id") is not None
        }
    return True, {}


def _citation_ids(finding: dict[str, Any]) -> list[str]:
    raw_ids = finding.get("evidence_ids")
    if not isinstance(raw_ids, list):
        return []
    return [str(item) for item in raw_ids]


def _citation_integrity(
    sample: dict[str, Any],
    findings: list[dict[str, Any]],
) -> str:
    evidence_present, evidence = _evidence_items(sample)
    sources_present, sources = _sources(sample)
    direct_source_citations = all(
        citation_id in sources
        for finding in findings
        for citation_id in _citation_ids(finding)
    )
    if not evidence_present and not (sources_present and direct_source_citations):
        return "not available: no saved evidence ledger or resolvable source citations"

    known_ids = set(evidence) | set(sources)
    valid_findings = sum(
        bool(citation_ids) and all(citation_id in known_ids for citation_id in citation_ids)
        for finding in findings
        for citation_ids in [_citation_ids(finding)]
    )
    if not findings:
        return "0 findings; broken-citation rate not applicable"
    broken_rate = (len(findings) - valid_findings) / len(findings)
    return (
        f"{valid_findings}/{len(findings)} findings resolve; "
        f"broken-citation rate {broken_rate:.0%}"
    )


def _support_distribution(findings: list[dict[str, Any]]) -> str:
    counts = {
        marker: sum(finding.get("support") == marker for finding in findings)
        for marker in _SUPPORT_MARKERS
    }
    missing = sum("support" not in finding for finding in findings)
    if not findings or not any("support" in finding for finding in findings):
        return "not available: no support markers are saved"
    distribution = ", ".join(f"{marker}={counts[marker]}" for marker in _SUPPORT_MARKERS)
    suffix = f", missing={missing}" if missing else ""
    return f"{distribution}{suffix} (pipeline self-check, not independent)"


def _domain_count(sample: dict[str, Any]) -> str:
    sources_present, sources = _sources(sample)
    if not sources_present:
        return "not available: no sources_used/sources field"
    domains = [source.get("domain") for source in sources.values()]
    if any(not isinstance(domain, str) or not domain.strip() for domain in domains):
        return "not available: one or more sources lack a domain"
    return str(len({domain.casefold() for domain in domains}))


def _snippet_share(sample: dict[str, Any]) -> str:
    sources_present, sources = _sources(sample)
    if not sources_present:
        return "not available: no sources_used/sources field"
    if not sources:
        return "not available: no saved sources"
    if any(not isinstance(source.get("snippet_only"), bool) for source in sources.values()):
        return "not available: one or more sources lack snippet_only"
    snippet_count = sum(source["snippet_only"] for source in sources.values())
    return f"{snippet_count}/{len(sources)} ({snippet_count / len(sources):.0%})"


def _source_urls_for_finding(
    finding: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> str:
    urls: list[str] = []
    for citation_id in _citation_ids(finding):
        source_id = citation_id if citation_id in sources else None
        evidence_item = evidence.get(citation_id)
        if source_id is None and evidence_item is not None:
            linked_source = evidence_item.get("source_id")
            if isinstance(linked_source, str) and linked_source in sources:
                source_id = linked_source
        source = sources.get(source_id) if source_id is not None else None
        url = source.get("url") if source is not None else None
        if isinstance(url, str) and url not in urls:
            urls.append(url)
    return "<br>".join(_markdown(url) for url in urls) or (
        "not available: no saved evidence-to-source mapping"
    )


def candidate_names(findings: Iterable[dict[str, Any]]) -> list[str]:
    """Return unique capitalized words found in finding statements."""
    names: list[str] = []
    seen: set[str] = set()
    for finding in findings:
        statement = finding.get("statement")
        if not isinstance(statement, str):
            continue
        for match in _CANDIDATE_RE.finditer(statement):
            name = match.group()
            key = name.casefold()
            if key not in seen:
                names.append(name)
                seen.add(key)
    return names


def _summary_row(path: Path, sample: dict[str, Any]) -> list[str]:
    raw_findings = sample.get("key_findings")
    findings = (
        [finding for finding in raw_findings if isinstance(finding, dict)]
        if isinstance(raw_findings, list)
        else []
    )
    return [
        _markdown(path.stem),
        str(len(findings)),
        _markdown(_citation_integrity(sample, findings)),
        _markdown(_support_distribution(findings)),
        _markdown(_domain_count(sample)),
        _markdown(_snippet_share(sample)),
    ]


def build_review_sheet(
    paths: Iterable[Path],
    *,
    include_candidates: bool = False,
) -> tuple[str, str, list[tuple[str, list[str]]]]:
    """Load samples and return markdown, console summary, and candidate lists."""
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        try:
            sample = json.loads(path.read_text(encoding="utf-8-sig"))
        except OSError as error:
            raise ValueError(f"cannot read {path}: {error}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}: {error}") from error
        if not isinstance(sample, dict):
            raise ValueError(f"sample root must be a JSON object: {path}")
        loaded.append((path, sample))

    summary_rows = [(path, sample, _summary_row(path, sample)) for path, sample in loaded]
    headers = [
        "Sample",
        "Findings",
        "Citation integrity",
        "Support markers",
        "Distinct domains",
        "Snippet-only share",
    ]
    summary_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    summary_lines.extend(
        "| " + " | ".join(row) + " |" for _path, _sample, row in summary_rows
    )

    sheet_lines = [
        "# Manual answer review sheet",
        "",
        "Pipeline support markers are self-reported, not independent human judgments.",
        "",
        "## Offline summary",
        "",
        *summary_lines,
        "",
    ]
    printed_candidates: list[tuple[str, list[str]]] = []
    if include_candidates:
        sheet_lines.extend(["## Candidate-name review aids", ""])
        for path, sample, _row in summary_rows:
            raw_findings = sample.get("key_findings")
            findings = (
                [finding for finding in raw_findings if isinstance(finding, dict)]
                if isinstance(raw_findings, list)
                else []
            )
            names = candidate_names(findings)
            printed_candidates.append((path.stem, names))
            sheet_lines.append(
                f"- **{_markdown(path.stem)}:** "
                f"{', '.join(_markdown(name) for name in names) or 'none found'}"
            )
        sheet_lines.append("")

    sheet_lines.extend(
        [
            "## Finding-level human review",
            "",
            "| Sample | Finding # | Finding text | Cited evidence IDs | Pipeline support | "
            "Cited source URLs | Human verdict (Supported / Partly / Not supported / "
            "Cannot check) | Note |",
            "|---|---:|---|---|---|---|---|---|",
        ]
    )
    for path, sample, _row in summary_rows:
        raw_findings = sample.get("key_findings")
        findings = (
            [finding for finding in raw_findings if isinstance(finding, dict)]
            if isinstance(raw_findings, list)
            else []
        )
        evidence_present, evidence = _evidence_items(sample)
        sources_present, sources = _sources(sample)
        for number, finding in enumerate(findings, 1):
            evidence_ids = _citation_ids(finding)
            urls = (
                _source_urls_for_finding(finding, evidence, sources)
                if evidence_present or sources_present
                else "not available: no evidence or source data saved"
            )
            sheet_lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown(path.stem),
                        str(number),
                        _markdown(finding.get("statement", "")),
                        _markdown(", ".join(evidence_ids)),
                        _markdown(finding.get("support", "not available")),
                        urls,
                        "",
                        "",
                    ]
                )
                + " |"
            )
    sheet_lines.append("")

    console_summary = "\n".join(summary_lines)
    if include_candidates:
        candidate_lines = [
            f"Candidates {sample}: {', '.join(names) if names else 'none found'}"
            for sample, names in printed_candidates
        ]
        console_summary += "\n" + "\n".join(candidate_lines)
    return "\n".join(sheet_lines), console_summary, printed_candidates


def _default_samples() -> list[Path]:
    return sorted(path for path in Path("samples").glob("*.json") if path.stem != "nonsense")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="sample JSON paths")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evals/review_sheet.md"),
        help="output markdown path (default: evals/review_sheet.md)",
    )
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="list capitalized words in findings as candidate-name review aids",
    )
    args = parser.parse_args(argv)
    paths = args.files or _default_samples()
    if not paths:
        parser.error("no sample JSON files found")
    try:
        sheet, summary, _candidate_rows = build_review_sheet(
            paths,
            include_candidates=args.candidates,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(sheet, encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    sys.stdout.write(summary + "\n")
    sys.stdout.write(f"Wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
