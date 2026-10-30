"""Seeded fault injection for search and fetch tools."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

from research_agent.errors import ToolFailure
from research_agent.schemas import (
    ErrorClass,
    FetchResult,
    FetchStatus,
    SearchArgs,
    SearchResult,
    ToolError,
)
from research_agent.tools.base import SearchProvider

_FAULT_NAMES = ("timeout", "http_403", "malformed", "empty")


class FaultInjector:
    """Parse fault probabilities and count each injected fault."""

    def __init__(self, spec: str, seed: int | None = None) -> None:
        self.probabilities: dict[str, float] = {}
        self.counts = {name: 0 for name in _FAULT_NAMES}
        self._random = random.Random(seed)
        for part in spec.split(",") if spec.strip() else ():
            name, separator, raw_probability = part.strip().partition(":")
            if not separator or name not in _FAULT_NAMES:
                raise ValueError(f"unknown or malformed fault: {part!r}")
            if name in self.probabilities:
                raise ValueError(f"duplicate fault: {name}")
            try:
                probability = float(raw_probability)
            except ValueError as err:
                raise ValueError(f"invalid probability for {name}") from err
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"probability for {name} must be between 0 and 1")
            self.probabilities[name] = probability

    def inject(self, name: str) -> bool:
        """Return whether the named fault fires and increment its count."""
        if name not in _FAULT_NAMES:
            raise ValueError(f"unknown fault: {name}")
        if self._random.random() >= self.probabilities.get(name, 0.0):
            return False
        self.counts[name] += 1
        return True

    def next_fault(self) -> str | None:
        """Choose at most one configured fault for an operation."""
        for name in self.probabilities:
            if self.inject(name):
                return name
        return None


def _failure(tool: str, error_class: ErrorClass, message: str) -> ToolFailure:
    return ToolFailure(ToolError(tool=tool, error_class=error_class, message=message))


def wrap_search_provider(provider: SearchProvider, injector: FaultInjector) -> SearchProvider:
    """Return a provider wrapper that injects one configured fault per call."""
    class WrappedSearchProvider:
        name = provider.name

        @property
        def last_dropped_count(self) -> int:
            return int(getattr(provider, "last_dropped_count", 0))

        def search(self, args: SearchArgs) -> list[SearchResult]:
            fault = injector.next_fault()
            if fault == "timeout":
                raise _failure(self.name, ErrorClass.TRANSIENT, "injected timeout")
            if fault == "http_403":
                raise _failure(self.name, ErrorClass.CLIENT, "injected HTTP 403")
            if fault == "malformed":
                raise _failure(self.name, ErrorClass.CONTENT, "injected malformed response")
            if fault == "empty":
                return []
            return provider.search(args)

    return WrappedSearchProvider()


def wrap_fetch(
    fetch_fn: Callable[..., FetchResult], injector: FaultInjector
) -> Callable[..., FetchResult]:
    """Return a fetch wrapper that injects one configured fault per call."""
    def wrapped(url: str, *args: Any, **kwargs: Any) -> FetchResult:  # noqa: ANN401
        fault = injector.next_fault()
        if fault == "timeout":
            raise _failure("fetch", ErrorClass.TRANSIENT, "injected timeout")
        if fault == "http_403":
            return FetchResult(url=url, status=FetchStatus.BLOCKED, http_status=403)
        if fault == "malformed":
            raise _failure("fetch", ErrorClass.CONTENT, "injected malformed response")
        if fault == "empty":
            return FetchResult(url=url, status=FetchStatus.EMPTY)
        return fetch_fn(url, *args, **kwargs)

    return wrapped