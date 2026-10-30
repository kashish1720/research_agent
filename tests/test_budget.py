"""Tests for resource and wall-time accounting."""

from __future__ import annotations

import pytest

import research_agent.budget as budget_module
from research_agent.budget import Budget
from research_agent.config import Settings
from research_agent.errors import BudgetExceeded


def _settings() -> Settings:
    return Settings(
        MAX_SEARCHES=2,
        MAX_FETCHES=1,
        MAX_LLM_CALLS=2,
        MAX_TOTAL_TOKENS=5,
        WALL_TIME_S=3,
    )


def test_search_budget_allows_limit_then_rejects_next() -> None:
    budget = Budget(_settings())
    assert budget.can_search()

    budget.consume_search()
    budget.consume_search()

    assert not budget.can_search()
    with pytest.raises(BudgetExceeded) as exc_info:
        budget.consume_search()
    assert exc_info.value.resource == "searches"
    assert budget.searches == 2


def test_fetch_budget_allows_limit_then_rejects_next() -> None:
    budget = Budget(_settings())
    budget.consume_fetch()

    assert not budget.can_fetch()
    with pytest.raises(BudgetExceeded) as exc_info:
        budget.consume_fetch()
    assert exc_info.value.resource == "fetches"
    assert budget.fetches == 1


def test_llm_call_limit_is_checked_before_increment() -> None:
    budget = Budget(_settings())
    budget.consume_llm(1, 1)
    budget.consume_llm(0, 0)

    assert not budget.can_llm()
    with pytest.raises(BudgetExceeded) as exc_info:
        budget.consume_llm(0, 0)
    assert exc_info.value.resource == "llm_calls"
    assert budget.llm_calls == 2


def test_token_limit_is_checked_without_partial_mutation() -> None:
    budget = Budget(_settings())
    budget.consume_llm(2, 3)

    with pytest.raises(BudgetExceeded) as exc_info:
        budget.consume_llm(1, 0)
    assert exc_info.value.resource == "tokens"
    assert budget.llm_calls == 1
    assert budget.input_tokens == 2
    assert budget.output_tokens == 3


def test_negative_token_count_is_rejected() -> None:
    budget = Budget(_settings())
    with pytest.raises(ValueError, match="non-negative"):
        budget.consume_llm(-1, 0)
    assert budget.llm_calls == 0


def test_time_clock_and_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [10.0]
    monkeypatch.setattr(budget_module.time, "monotonic", lambda: now[0])
    budget = Budget(_settings())
    now[0] = 12.5

    assert budget.elapsed() == pytest.approx(2.5)
    assert budget.time_left() == pytest.approx(0.5)
    budget.consume_search()
    budget.tool_retries = 1
    assert budget.snapshot() == {
        "searches": 1,
        "fetches": 0,
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tool_retries": 1,
        "elapsed_s": pytest.approx(2.5),
    }


def test_check_time_raises_only_after_wall_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [10.0]
    monkeypatch.setattr(budget_module.time, "monotonic", lambda: now[0])
    budget = Budget(_settings())
    now[0] = 13.0
    budget.check_time()
    now[0] = 13.01

    with pytest.raises(BudgetExceeded) as exc_info:
        budget.check_time()
    assert exc_info.value.resource == "time"


def test_start_restarts_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [10.0]
    monkeypatch.setattr(budget_module.time, "monotonic", lambda: now[0])
    budget = Budget(_settings())
    now[0] = 15.0
    budget.start()

    assert budget.elapsed() == 0.0