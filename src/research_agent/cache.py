"""Small JSON-backed record/replay cache for successful calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from research_agent.budget import Budget
from research_agent.schemas import SearchArgs, SearchResult
from research_agent.tools.base import SearchProvider
from research_agent.tracing import Tracer


def cache_key(inputs: dict[str, object]) -> str:
    """Hash canonical JSON call inputs."""
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DiskCache:
    """Read and write JSON values beneath one configured cache directory."""

    def __init__(self, cache_dir: str | Path, *, enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

    def load(self, kind: str, inputs: dict[str, object]) -> tuple[bool, Any]:  # noqa: ANN401
        if not self.enabled:
            return False, None
        path = self._path(kind, inputs)
        if not path.exists():
            return False, None
        return True, json.loads(path.read_text(encoding="utf-8"))

    def store(self, kind: str, inputs: dict[str, object], value: object) -> None:
        if not self.enabled:
            return
        path = self._path(kind, inputs)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )

    def _path(self, kind: str, inputs: dict[str, object]) -> Path:
        if not kind or any(part in {".", ".."} for part in Path(kind).parts):
            raise ValueError("cache kind must be a non-empty relative path")
        return self.cache_dir / kind / f"{cache_key(inputs)}.json"


class CachedSearchProvider:
    """Cache successful normalized provider responses on disk."""

    def __init__(self, provider: SearchProvider, cache: DiskCache) -> None:
        self.provider = provider
        self.cache = cache
        self.name = provider.name
        self.last_dropped_count = 0
        self.tracer: Tracer | None = None
        self.budget: Budget | None = None

    def set_run_context(self, tracer: Tracer, budget: Budget) -> None:
        self.tracer = tracer
        self.budget = budget

    def search(self, args: SearchArgs) -> list[SearchResult]:
        inputs: dict[str, object] = {
            "provider": self.name,
            "query": args.query,
            "max_results": args.max_results,
            "depth": args.depth,
        }
        found, value = self.cache.load("search", inputs)
        if found:
            if self.tracer is not None:
                self.tracer.event("note", f"search:{self.name}", cache_hit=True)
            if self.budget is not None and self.budget.searches > 0:
                self.budget.searches -= 1
            self.last_dropped_count = 0
            return [SearchResult.model_validate(item) for item in value]

        results = self.provider.search(args)
        self.last_dropped_count = int(getattr(self.provider, "last_dropped_count", 0))
        self.cache.store(
            "search",
            inputs,
            [result.model_dump(mode="json") for result in results],
        )
        return results
