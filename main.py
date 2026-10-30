"""Command-line entry point for the research agent."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from research_agent.config import load_settings
from research_agent.pipeline import ConfigError, run_pipeline
from research_agent.render import render_answer
from research_agent.schemas import FinalAnswer, RunStatus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research a question with cited evidence.")
    parser.add_argument("question", nargs="?", help="Research question")
    parser.add_argument("--question-file", help="Read the question from a UTF-8 text file")
    parser.add_argument("--json", action="store_true", help="Write only FinalAnswer JSON")
    parser.add_argument("--trace", action="store_true", help="Print per-node trace summary")
    parser.add_argument("--save", metavar="PATH", help="Also save FinalAnswer JSON to PATH")
    parser.add_argument("--faults", help="Enable seeded fault injection for this run")
    parser.add_argument("--no-cache", action="store_true", help="Disable the disk cache")
    parser.add_argument("--debug", action="store_true", help="Show stack traces for crashes")
    return parser


def _read_question(args: argparse.Namespace) -> str:
    if bool(args.question) == bool(args.question_file):
        raise ConfigError("Provide exactly one positional question or --question-file.")
    if args.question_file:
        try:
            question = Path(args.question_file).read_text(encoding="utf-8").strip()
        except OSError as error:
            raise ConfigError("Unable to read --question-file.") from error
    else:
        question = args.question.strip()
    if not question:
        raise ConfigError("The research question must not be empty.")
    return question


def _trace_summary(run_id: str, log_dir: str) -> str:
    trace_path = Path(log_dir) / f"{run_id}.jsonl"
    try:
        events = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return "Trace summary unavailable."
    for event in reversed(events):
        details = event.get("details")
        summary = details.get("trace_summary") if isinstance(details, dict) else None
        if isinstance(summary, dict):
            rows = ["Node | time ms | tokens in/out | tool calls | retries | failures"]
            for node, values in sorted(summary.items()):
                rows.append(
                    f"{node} | {values['duration_ms']:.1f} | "
                    f"{values['tokens_in']}/{values['tokens_out']} | "
                    f"{values['tool_calls']} | {values['retries']} | {values['failures']}"
                )
            return "\n".join(rows)
    return "Trace summary unavailable."


def _write_answer(answer: FinalAnswer, *, as_json: bool, output: TextIO) -> None:
    if as_json:
        output.write(
            json.dumps(answer.model_dump(mode="json"), ensure_ascii=False) + "\n"
        )
    else:
        render_answer(answer, output=output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        question = _read_question(args)
        settings = load_settings()
        if args.no_cache:
            settings.USE_CACHE = False
        if args.faults is not None:
            settings.FAULT_INJECT = args.faults
        answer = run_pipeline(question, settings=settings)
        if args.save:
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(
                json.dumps(answer.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    except ConfigError as error:
        sys.stderr.write(f"configuration error: {error}\n")
        return 1
    except (OSError, ValueError) as error:
        sys.stderr.write(f"configuration error: {type(error).__name__}\n")
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        return 1
    except Exception as error:
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        else:
            sys.stderr.write(f"research run failed ({type(error).__name__})\n")
        return 1

    _write_answer(answer, as_json=args.json, output=sys.stdout)
    if args.trace:
        sys.stderr.write(_trace_summary(answer.run_id, settings.LOG_DIR) + "\n")
    return 2 if answer.status is RunStatus.FAILED_GRACEFULLY else 0


if __name__ == "__main__":
    sys.exit(main())
