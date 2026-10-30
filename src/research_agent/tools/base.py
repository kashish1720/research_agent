"""Shared contracts for tool implementations."""

from __future__ import annotations

from typing import Protocol

from research_agent.schemas import SearchArgs, SearchResult


class SearchProvider(Protocol):
    """A provider that returns normalized search results."""

    name: str

    def search(self, args: SearchArgs) -> list[SearchResult]:
        """Search for ``args.query`` and return normalized results."""
