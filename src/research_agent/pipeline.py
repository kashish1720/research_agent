"""Sequential plain-Python orchestration for a research run."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import replace
from datetime import date
from typing import Any

from research_agent.budget import Budget
from research_agent.config import Settings, load_settings
from research_agent.deps import Deps, RunContext, build_default_deps
from research_agent.errors import BudgetExceeded
from research_agent.nodes.extract import make_extract_node
from research_agent.nodes.finalize import make_finalize_node
from research_agent.nodes.plan import make_plan_node
from research_agent.nodes.search import make_search_node
from research_agent.nodes.select_fetch import make_select_fetch_node
from research_agent.nodes.sufficiency import compute_coverage, make_sufficiency_node
from research_agent.nodes.synthesize import make_synthesize_node
from research_agent.nodes.verify import make_verify_node
from research_agent.schemas import AgentState, FinalAnswer, Reason
from research_agent.tools.faults import FaultInjector, wrap_search_provider
from research_agent.tools.url_safety import UrlAllowlist
from research_agent.tracing import Tracer


class ConfigError(Exception):
    """Raised when settings or required provider configuration is invalid."""


def _append_reason(reasons: list[Reason], reason: Reason) -> list[Reason]:
    if reason not in reasons:
        reasons.append(reason)
    return reasons


def _run_context_fallback(settings: Settings) -> RunContext:
    """Create a context if per-run setup itself fails, allowing graceful finalization."""
    run_id = f"failed-{date.today().strftime('%Y%m%d')}"
    return RunContext(
        run_id=run_id,
        budget=Budget(settings),
        tracer=Tracer(run_id, settings.LOG_DIR),
        allowlist=UrlAllowlist(),
        fetch_fn=lambda url: _unavailable_fetch(url),
    )


def _unavailable_fetch(url: str):
    from research_agent.schemas import FetchResult, FetchStatus

    return FetchResult(url=url, status=FetchStatus.EMPTY)


def run_pipeline(
    question: str,
    deps: Deps | None = None,
    settings: Settings | None = None,
) -> FinalAnswer:
    """Run the bounded research stages sequentially and always finalize."""
    try:
        active_settings = settings or (deps.settings if deps is not None else load_settings())
        active_deps = deps or build_default_deps(active_settings)
        if settings is not None and active_deps.settings is not settings:
            active_deps = replace(active_deps, settings=settings)
    except (ValueError, OSError) as error:
        raise ConfigError(f"Research configuration is invalid ({type(error).__name__})") from error

    try:
        run = active_deps.new_run(question)
    except Exception as error:
        try:
            run = _run_context_fallback(active_settings)
            run.tracer.event(
                "note",
                "pipeline",
                details={"action": "run_setup_failed", "error_type": type(error).__name__},
            )
        except Exception as setup_error:
            raise ConfigError(
                f"Unable to initialize the run context ({type(setup_error).__name__})"
            ) from setup_error

    if active_settings.FAULT_INJECT.strip() and active_deps.search_providers:
        injector = FaultInjector(
            active_settings.FAULT_INJECT,
            seed=int(run.run_id[-6:], 16) if run.run_id[-6:].isalnum() else None,
        )
        active_deps = replace(
            active_deps,
            search_providers=[
                wrap_search_provider(provider, injector)
                for provider in active_deps.search_providers
            ],
        )

    state = AgentState(
        question=question,
        today=date.today().isoformat(),
    )
    plan_node = make_plan_node(active_deps, run)
    search_node = make_search_node(active_deps, run)
    select_fetch_node = make_select_fetch_node(active_deps, run)
    extract_node = make_extract_node(active_deps, run)
    sufficiency_node = make_sufficiency_node(active_deps, run)
    synthesize_node = make_synthesize_node(active_deps, run)
    verify_node = make_verify_node(active_deps, run)
    finalize_node = make_finalize_node(active_deps, run)

    def call_node(
        name: str,
        node: Callable[[AgentState], dict[str, Any]],
        current: AgentState,
        *,
        check_time: bool = True,
    ) -> AgentState:
        try:
            if check_time and name != "verify":
                run.budget.check_time()
            span = run.tracer.node_span(name) if name == "finalize" else nullcontext()
            with span:
                update = node(current)
            return current.model_copy(update=update)
        except BudgetExceeded as exhausted:
            reasons = _append_reason(list(current.reasons), Reason.BUDGET_EXHAUSTED)
            run.tracer.event(
                "budget",
                name,
                details={"resource": exhausted.resource},
            )
            changes: dict[str, Any] = {"reasons": reasons}
            if name == "sufficiency" and current.plan is not None:
                coverage = compute_coverage(current.plan, current.evidence)
                coverage_ratio = coverage.covered / coverage.total if coverage.total else 0
                if coverage_ratio >= active_settings.MIN_COVERAGE_TO_ANSWER:
                    changes["route"] = "synthesize"
                    if coverage.covered < coverage.total:
                        changes["reasons"] = _append_reason(
                            reasons,
                            Reason.UNCOVERED_SUBQUESTIONS,
                        )
                else:
                    changes["route"] = "insufficient"
                    changes["reasons"] = _append_reason(
                        reasons,
                        Reason.INSUFFICIENT_EVIDENCE,
                    )
                return current.model_copy(update=changes)
            changes.update(
                abort=True,
                errors=[
                    *current.errors,
                    f"{name} stopped because the {exhausted.resource} budget was exhausted",
                ],
            )
            return current.model_copy(update=changes)
        except Exception as error:
            message = f"{name} failed ({type(error).__name__})"
            reasons = _append_reason(list(current.reasons), Reason.NODE_ERROR)
            run.tracer.event(
                "note",
                name,
                details={"action": "node_failed", "error_type": type(error).__name__},
            )
            return current.model_copy(
                update={
                    "errors": [*current.errors, message],
                    "reasons": reasons,
                    "abort": True,
                }
            )

    state = call_node("plan", plan_node, state)
    if state.abort:
        state = call_node("finalize", finalize_node, state, check_time=False)
        return state.final  # type: ignore[return-value]

    refinement_count = 0
    while True:
        state = call_node("search", search_node, state)
        if state.abort:
            break
        state = call_node("select_fetch", select_fetch_node, state)
        if state.abort:
            break
        state = call_node("extract", extract_node, state)
        if state.abort:
            break

        if run.budget.time_left() <= 0:
            state = state.model_copy(
                update={
                    "reasons": _append_reason(
                        list(state.reasons),
                        Reason.BUDGET_EXHAUSTED,
                    )
                }
            )
            coverage = compute_coverage(state.plan, state.evidence)  # type: ignore[arg-type]
            coverage_ratio = coverage.covered / coverage.total if coverage.total else 0
            if coverage_ratio >= active_settings.MIN_COVERAGE_TO_ANSWER:
                state = state.model_copy(
                    update={
                        "route": "synthesize",
                        "reasons": _append_reason(
                            list(state.reasons),
                            Reason.UNCOVERED_SUBQUESTIONS,
                        )
                        if coverage.covered < coverage.total
                        else state.reasons,
                    }
                )
            else:
                state = state.model_copy(
                    update={
                        "route": "insufficient",
                        "reasons": _append_reason(
                            list(state.reasons),
                            Reason.INSUFFICIENT_EVIDENCE,
                        ),
                    }
                )
        else:
            state = call_node("sufficiency", sufficiency_node, state)

        if state.abort or state.route == "insufficient":
            break
        if state.route == "refine":
            refinement_count += 1
            if refinement_count > active_settings.MAX_REFINE_LOOPS:
                state = state.model_copy(
                    update={
                        "errors": [*state.errors, "Refinement iteration limit exceeded"],
                        "reasons": _append_reason(
                            list(state.reasons),
                            Reason.NODE_ERROR,
                        ),
                        "abort": True,
                    }
                )
                break
            continue
        if state.route == "synthesize":
            state = call_node(
                "synthesize",
                synthesize_node,
                state,
                check_time=run.budget.time_left() > 0,
            )
            if state.abort:
                break
            state = call_node("verify", verify_node, state, check_time=False)
            break
        state = state.model_copy(
            update={
                "errors": [*state.errors, "Sufficiency returned no valid route"],
                "reasons": _append_reason(list(state.reasons), Reason.NODE_ERROR),
                "abort": True,
            }
        )
        break

    state = call_node("finalize", finalize_node, state, check_time=False)
    return state.final  # type: ignore[return-value]
