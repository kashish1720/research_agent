"""Tests for structured runtime exceptions."""

from __future__ import annotations

import pytest

from research_agent.errors import BudgetExceeded, ToolFailure
from research_agent.schemas import ErrorClass, ToolError


def test_tool_failure_retains_structured_error() -> None:
    error = ToolError(tool="search", error_class=ErrorClass.CLIENT, message="bad request")

    failure = ToolFailure(error)

    assert failure.error is error
    assert str(failure) == "bad request"


@pytest.mark.parametrize("resource", ["searches", "fetches", "llm_calls", "tokens", "time"])
def test_budget_exceeded_retains_resource(resource: str) -> None:
    failure = BudgetExceeded(resource)  # type: ignore[arg-type]

    assert failure.resource == resource
    assert resource in str(failure)


def test_budget_exceeded_rejects_unknown_resource() -> None:
    with pytest.raises(ValueError, match="unknown budget resource"):
        BudgetExceeded("downloads")  # type: ignore[arg-type]