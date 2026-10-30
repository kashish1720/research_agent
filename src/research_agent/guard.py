"""Sequential validation, memoization, and retry policy for tools."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from research_agent.budget import Budget
from research_agent.errors import BudgetExceeded, ToolFailure
from research_agent.schemas import ErrorClass, ToolError
from research_agent.tracing import Tracer

ArgsT = TypeVar("ArgsT", bound=BaseModel)
ResultT = TypeVar("ResultT")
RepairFn = Callable[[str, object], object]


class Memo:
	"""Per-run cache of successful calls and permanent failures."""

	def __init__(self) -> None:
		self._successes: dict[tuple[str, str], object] = {}
		self._failures: dict[tuple[str, str], ToolError] = {}

	def success(self, key: tuple[str, str]) -> tuple[bool, object | None]:
		if key not in self._successes:
			return False, None
		return True, self._successes[key]

	def failure(self, key: tuple[str, str]) -> ToolError | None:
		return self._failures.get(key)

	def store_success(self, key: tuple[str, str], result: object) -> None:
		self._successes[key] = result

	def store_failure(self, key: tuple[str, str], error: ToolError) -> None:
		self._failures[key] = error


def _canonical_args(args: BaseModel) -> str:
	return json.dumps(
		args.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
	)


def _tool_error(tool: str, error_class: ErrorClass, message: str, attempt: int = 1) -> ToolFailure:
	return ToolFailure(
		ToolError(tool=tool, error_class=error_class, message=message, attempt=attempt)
	)


def _consume_budget(budget: Budget, budget_kind: str) -> None:
	if budget_kind == "search":
		budget.consume_search()
	elif budget_kind == "fetch":
		budget.consume_fetch()
	elif budget_kind == "llm":
		budget.consume_llm(0, 0)
	elif budget_kind != "none":
		raise ValueError(f"unknown budget kind: {budget_kind}")


def guarded_call(
	*,
	tool: str,
	fn: Callable[[ArgsT], ResultT],
	args_model: type[ArgsT],
	raw_args: object,
	budget: Budget,
	tracer: Tracer,
	budget_kind: str,
	repair_fn: RepairFn | None = None,
	sleep_fn: Callable[[float], None] = time.sleep,
	memo: Memo | None = None,
) -> ResultT:
	"""Validate and execute a tool call under budget, retry, and memo policies."""
	corrections = 0
	current_raw_args = raw_args
	while True:
		try:
			validated_args = args_model.model_validate(current_raw_args)
			break
		except ValidationError as err:
			max_corrections = budget.settings.MAX_CORRECTIONS
			if repair_fn is None or corrections >= max_corrections:
				raise _tool_error(
					tool,
					ErrorClass.SCHEMA,
					f"invalid arguments after {corrections} corrections",
				) from err
			corrections += 1
			try:
				current_raw_args = repair_fn(str(err), current_raw_args)
			except Exception as repair_error:
				raise _tool_error(
					tool,
					ErrorClass.SCHEMA,
					f"argument repair failed ({type(repair_error).__name__})",
				) from repair_error

	memo_key = (tool, _canonical_args(validated_args))
	active_memo = memo if memo is not None else Memo()
	cached, cached_result = active_memo.success(memo_key)
	if cached:
		tracer.event("note", tool, details={"memo": "success"})
		return cached_result  # type: ignore[return-value]
	cached_error = active_memo.failure(memo_key)
	if cached_error is not None:
		tracer.event(
			"note",
			tool,
			details={"memo": "permanent_failure", "error_class": cached_error.error_class.value},
		)
		raise ToolFailure(cached_error)

	max_retries = budget.settings.MAX_RETRIES
	for attempt in range(1, max_retries + 2):
		try:
			_consume_budget(budget, budget_kind)
		except BudgetExceeded as err:
			tracer.event("budget", tool, details={"resource": err.resource})
			raise

		tracer.event("tool_call", tool, attempt=attempt)
		try:
			result = fn(validated_args)
		except ToolFailure as failure:
			failure_error = failure.error.model_copy(update={"attempt": attempt})
			if failure_error.error_class is not ErrorClass.TRANSIENT:
				active_memo.store_failure(memo_key, failure_error)
				tracer.event(
					"error",
					tool,
					attempt=attempt,
					details={"error_class": failure_error.error_class.value},
				)
				raise ToolFailure(failure_error) from failure
			tracer.event(
				"error",
				tool,
				attempt=attempt,
				details={"error_class": failure_error.error_class.value},
			)
			if attempt > max_retries:
				raise ToolFailure(failure_error) from failure
			budget.tool_retries += 1
			delay = budget.settings.BACKOFF_BASE_S * (2 ** (attempt - 1))
			delay += random.uniform(0.0, delay * 0.1)
			tracer.event(
				"retry",
				tool,
				attempt=attempt,
				details={"delay_s": delay},
			)
			sleep_fn(delay)
		except Exception as err:
			mapped_error = ToolError(
				tool=tool,
				error_class=ErrorClass.UNKNOWN,
				message=f"unexpected {type(err).__name__}: {err}",
				attempt=attempt,
			)
			active_memo.store_failure(memo_key, mapped_error)
			tracer.event(
				"error",
				tool,
				attempt=attempt,
				details={"error_class": ErrorClass.UNKNOWN.value},
			)
			raise ToolFailure(mapped_error) from err
		else:
			active_memo.store_success(memo_key, result)
			return result

	raise AssertionError("retry loop exited unexpectedly")

