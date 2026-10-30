"""Sequential resource and wall-time budget tracking."""

from __future__ import annotations

import time
from typing import TypedDict

from research_agent.config import Settings
from research_agent.errors import BudgetExceeded


class BudgetSnapshot(TypedDict):
	searches: int
	fetches: int
	llm_calls: int
	input_tokens: int
	output_tokens: int
	tool_retries: int
	elapsed_s: float


class Budget:
	"""Track per-run resource use against the configured limits."""

	def __init__(self, settings: Settings) -> None:
		self.settings = settings
		self.searches = 0
		self.fetches = 0
		self.llm_calls = 0
		self.input_tokens = 0
		self.output_tokens = 0
		self.tool_retries = 0
		self._started_at: float | None = None
		self.start()

	def start(self) -> None:
		"""Start or restart the wall-time clock."""
		self._started_at = time.monotonic()

	def elapsed(self) -> float:
		"""Return elapsed seconds since ``start``."""
		if self._started_at is None:
			return 0.0
		return max(0.0, time.monotonic() - self._started_at)

	def time_left(self) -> float:
		"""Return the non-negative remaining wall-time budget."""
		return max(0.0, float(self.settings.WALL_TIME_S) - self.elapsed())

	def check_time(self) -> None:
		"""Raise after the run has exceeded its wall-time limit."""
		if self.elapsed() > self.settings.WALL_TIME_S:
			raise BudgetExceeded("time")

	def can_search(self) -> bool:
		return self.searches < self.settings.MAX_SEARCHES

	def can_fetch(self) -> bool:
		return self.fetches < self.settings.MAX_FETCHES

	def can_llm(self) -> bool:
		total_tokens = self.input_tokens + self.output_tokens
		return (
			self.llm_calls < self.settings.MAX_LLM_CALLS
			and total_tokens < self.settings.MAX_TOTAL_TOKENS
		)

	def consume_search(self) -> None:
		if not self.can_search():
			raise BudgetExceeded("searches")
		self.searches += 1

	def consume_fetch(self) -> None:
		if not self.can_fetch():
			raise BudgetExceeded("fetches")
		self.fetches += 1

	def consume_llm(self, tokens_in: int, tokens_out: int) -> None:
		if tokens_in < 0 or tokens_out < 0:
			raise ValueError("token counts must be non-negative")
		if self.llm_calls >= self.settings.MAX_LLM_CALLS:
			raise BudgetExceeded("llm_calls")
		if self.input_tokens + self.output_tokens + tokens_in + tokens_out > (
			self.settings.MAX_TOTAL_TOKENS
		):
			raise BudgetExceeded("tokens")
		self.llm_calls += 1
		self.input_tokens += tokens_in
		self.output_tokens += tokens_out

	def snapshot(self) -> BudgetSnapshot:
		"""Return a stable snapshot of counters and elapsed time."""
		return {
			"searches": self.searches,
			"fetches": self.fetches,
			"llm_calls": self.llm_calls,
			"input_tokens": self.input_tokens,
			"output_tokens": self.output_tokens,
			"tool_retries": self.tool_retries,
			"elapsed_s": self.elapsed(),
		}

