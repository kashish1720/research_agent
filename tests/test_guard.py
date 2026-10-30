"""Tests for argument validation, memoization, retries, and guarded calls."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from research_agent.budget import Budget
from research_agent.config import Settings
from research_agent.errors import BudgetExceeded, ToolFailure
from research_agent.guard import Memo, guarded_call
from research_agent.schemas import ErrorClass, ToolError
from research_agent.tracing import Tracer


class QueryArgs(BaseModel):
    query: str


class IntegerArgs(BaseModel):
    query: int


def _runtime(tmp_path, **overrides):  # type: ignore[no-untyped-def]
    settings = Settings(**overrides)
    return Budget(settings), Tracer("guard-test", tmp_path)


def _transient(tool: str = "search") -> ToolFailure:
    return ToolFailure(
        ToolError(tool=tool, error_class=ErrorClass.TRANSIENT, message="temporary")
    )


def test_transient_twice_then_success_uses_exponential_delays(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    budget, tracer = _runtime(tmp_path, MAX_RETRIES=2, BACKOFF_BASE_S=0.5)
    monkeypatch.setattr("research_agent.guard.random.uniform", lambda _low, _high: 0.0)
    attempts = 0
    delays: list[float] = []

    def call(args: QueryArgs) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise _transient()
        return args.query.upper()

    result = guarded_call(
        tool="search",
        fn=call,
        args_model=QueryArgs,
        raw_args={"query": "hello"},
        budget=budget,
        tracer=tracer,
        budget_kind="search",
        sleep_fn=delays.append,
    )

    assert result == "HELLO"
    assert attempts == 3
    assert delays == [0.5, 1.0]
    assert budget.searches == 3
    assert budget.tool_retries == 2


def test_permanent_failure_is_not_retried(tmp_path) -> None:  # type: ignore[no-untyped-def]
    budget, tracer = _runtime(tmp_path)
    attempts = 0

    def call(_args: QueryArgs) -> None:
        nonlocal attempts
        attempts += 1
        raise ToolFailure(
            ToolError(tool="search", error_class=ErrorClass.CLIENT, message="denied")
        )

    with pytest.raises(ToolFailure) as exc_info:
        guarded_call(
            tool="search",
            fn=call,
            args_model=QueryArgs,
            raw_args={"query": "hello"},
            budget=budget,
            tracer=tracer,
            budget_kind="search",
        )
    assert exc_info.value.error.error_class is ErrorClass.CLIENT
    assert attempts == 1


def test_retry_exhaustion_raises_last_transient_failure(tmp_path) -> None:  # type: ignore[no-untyped-def]
    budget, tracer = _runtime(tmp_path, MAX_RETRIES=1, BACKOFF_BASE_S=0.1)
    attempts = 0

    def call(_args: QueryArgs) -> None:
        nonlocal attempts
        attempts += 1
        raise _transient()

    with pytest.raises(ToolFailure) as exc_info:
        guarded_call(
            tool="search",
            fn=call,
            args_model=QueryArgs,
            raw_args={"query": "hello"},
            budget=budget,
            tracer=tracer,
            budget_kind="search",
            sleep_fn=lambda _delay: None,
        )
    assert attempts == 2
    assert exc_info.value.error.attempt == 2
    assert budget.tool_retries == 1


def test_repair_succeeds_on_first_correction(tmp_path) -> None:  # type: ignore[no-untyped-def]
    budget, tracer = _runtime(tmp_path)
    repairs: list[object] = []

    def repair(error_text: str, previous: object) -> object:
        assert "query" in error_text
        repairs.append(previous)
        return {"query": "fixed"}

    result = guarded_call(
        tool="lookup",
        fn=lambda args: args.query,
        args_model=QueryArgs,
        raw_args={"wrong": "value"},
        budget=budget,
        tracer=tracer,
        budget_kind="none",
        repair_fn=repair,
    )

    assert result == "fixed"
    assert len(repairs) == 1


def test_repair_limit_raises_schema_failure(tmp_path) -> None:  # type: ignore[no-untyped-def]
    budget, tracer = _runtime(tmp_path, MAX_CORRECTIONS=2)
    repairs = 0

    def repair(_error_text: str, previous: object) -> object:
        nonlocal repairs
        repairs += 1
        return previous

    with pytest.raises(ToolFailure) as exc_info:
        guarded_call(
            tool="lookup",
            fn=lambda _args: pytest.fail("invalid arguments reached fn"),
            args_model=QueryArgs,
            raw_args={"wrong": "value"},
            budget=budget,
            tracer=tracer,
            budget_kind="none",
            repair_fn=repair,
        )
    assert repairs == 2
    assert exc_info.value.error.error_class is ErrorClass.SCHEMA


def test_repair_is_not_called_for_valid_arguments(tmp_path) -> None:  # type: ignore[no-untyped-def]
    budget, tracer = _runtime(tmp_path)
    called = False

    def repair(_error_text: str, _previous: object) -> object:
        nonlocal called
        called = True
        return {}

    assert guarded_call(
        tool="lookup",
        fn=lambda args: args.query,
        args_model=QueryArgs,
        raw_args={"query": "valid"},
        budget=budget,
        tracer=tracer,
        budget_kind="none",
        repair_fn=repair,
    ) == "valid"
    assert not called


def test_success_memo_hit_does_not_consume_budget(tmp_path) -> None:  # type: ignore[no-untyped-def]
    budget, tracer = _runtime(tmp_path)
    memo = Memo()
    calls = 0

    def call(_args: QueryArgs) -> str:
        nonlocal calls
        calls += 1
        return "cached"

    options = {
        "tool": "search",
        "fn": call,
        "args_model": QueryArgs,
        "raw_args": {"query": "same"},
        "budget": budget,
        "tracer": tracer,
        "budget_kind": "search",
        "memo": memo,
    }
    assert guarded_call(**options) == "cached"  # type: ignore[arg-type]
    assert guarded_call(**options) == "cached"  # type: ignore[arg-type]
    assert calls == 1
    assert budget.searches == 1
    assert tracer.events[-1]["event"] == "note"


def test_memoized_permanent_failure_short_circuits(tmp_path) -> None:  # type: ignore[no-untyped-def]
    budget, tracer = _runtime(tmp_path)
    memo = Memo()
    calls = 0

    def call(_args: QueryArgs) -> None:
        nonlocal calls
        calls += 1
        raise ToolFailure(
            ToolError(tool="search", error_class=ErrorClass.CONTENT, message="invalid")
        )

    for _ in range(2):
        with pytest.raises(ToolFailure):
            guarded_call(
                tool="search",
                fn=call,
                args_model=QueryArgs,
                raw_args={"query": "same"},
                budget=budget,
                tracer=tracer,
                budget_kind="search",
                memo=memo,
            )
    assert calls == 1
    assert budget.searches == 1


def test_budget_exceeded_propagates_before_tool_call(tmp_path) -> None:  # type: ignore[no-untyped-def]
    budget, tracer = _runtime(tmp_path, MAX_SEARCHES=1)
    budget.consume_search()
    calls = 0

    def call(_args: QueryArgs) -> None:
        nonlocal calls
        calls += 1

    with pytest.raises(BudgetExceeded) as exc_info:
        guarded_call(
            tool="search",
            fn=call,
            args_model=QueryArgs,
            raw_args={"query": "valid"},
            budget=budget,
            tracer=tracer,
            budget_kind="search",
        )
    assert exc_info.value.resource == "searches"
    assert calls == 0


def test_unexpected_value_error_maps_to_unknown(tmp_path) -> None:  # type: ignore[no-untyped-def]
    budget, tracer = _runtime(tmp_path)

    def call(_args: QueryArgs) -> None:
        raise ValueError("bad response")

    with pytest.raises(ToolFailure) as exc_info:
        guarded_call(
            tool="search",
            fn=call,
            args_model=QueryArgs,
            raw_args={"query": "valid"},
            budget=budget,
            tracer=tracer,
            budget_kind="none",
        )
    assert exc_info.value.error.error_class is ErrorClass.UNKNOWN
    assert "ValueError" in exc_info.value.error.message