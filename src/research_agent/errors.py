"""Custom exceptions and error classes."""

from __future__ import annotations

from typing import Literal

from research_agent.schemas import ToolError


class ToolFailure(Exception):
	"""A tool failure carrying its structured error details."""

	def __init__(self, error: ToolError) -> None:
		self.error = error
		super().__init__(error.message)


class BudgetExceeded(Exception):
	"""Raised when a configured run budget has been exhausted."""

	Resource = Literal["searches", "fetches", "llm_calls", "tokens", "time"]

	def __init__(self, resource: Resource) -> None:
		if resource not in {"searches", "fetches", "llm_calls", "tokens", "time"}:
			raise ValueError(f"unknown budget resource: {resource}")
		self.resource = resource
		super().__init__(f"budget exceeded: {resource}")

