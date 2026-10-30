"""Run only the live plan node and display its JSON and token usage."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import date

from research_agent.config import Settings, load_settings
from research_agent.deps import build_default_deps
from research_agent.nodes.plan import make_plan_node
from research_agent.schemas import AgentState


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one live planning request against Groq.")
    parser.add_argument("question", help="Research question to plan")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the disk cache for this run",
    )
    return parser.parse_args(argv)


def _apply_cli_options(settings: Settings, *, no_cache: bool) -> Settings:
    if no_cache:
        settings.USE_CACHE = False
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one live planning request against Groq.")
    args = _parse_args()

    settings = _apply_cli_options(load_settings(), no_cache=args.no_cache)
    settings.SEARCH_PROVIDERS = ""
    try:
        deps = build_default_deps(settings)
    except ValueError as error:
        parser.exit(2, f"error: {error}\n")

    run = deps.new_run(args.question)
    state = AgentState(question=args.question, today=date.today().isoformat())
    update = make_plan_node(deps, run)(state)
    plan = update["plan"]
    print(json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False))
    usage = run.budget.snapshot()
    print(
        "Token usage: "
        f"input={usage['input_tokens']} output={usage['output_tokens']} "
        f"model={deps.llm.model_for('cheap')}"
    )


if __name__ == "__main__":
    main()