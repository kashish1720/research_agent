"""Process-level dependencies and per-run execution context."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial

from research_agent.budget import Budget
from research_agent.cache import CachedSearchProvider, DiskCache
from research_agent.config import Settings, load_settings
from research_agent.filters import normalize_url
from research_agent.llm import GroqClient, LLMClient
from research_agent.schemas import FetchResult, FetchStatus
from research_agent.tools.base import SearchProvider
from research_agent.tools.ddgs_search import DDGSSearchProvider
from research_agent.tools.faults import FaultInjector, wrap_fetch
from research_agent.tools.fetch_page import fetch_page
from research_agent.tools.tavily_search import TavilySearchProvider
from research_agent.tools.url_safety import UrlAllowlist
from research_agent.tracing import Tracer

FetchFn = Callable[[str], FetchResult]
FetchFactory = Callable[[UrlAllowlist], FetchFn]


@dataclass
class RunContext:
	"""Resources scoped to a single research run."""

	run_id: str
	budget: Budget
	tracer: Tracer
	allowlist: UrlAllowlist
	fetch_fn: FetchFn


@dataclass
class Deps:
	"""Shared settings and process-level provider/client instances."""

	settings: Settings
	llm: LLMClient
	search_providers: list[SearchProvider]
	fetch_factory: FetchFactory

	def new_run(self, question: str) -> RunContext:
		"""Create isolated run state, tracing, allowlisting, and fetch wrapper."""
		del question
		run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(3)}"
		budget = Budget(self.settings)
		tracer = Tracer(run_id, self.settings.LOG_DIR)
		allowlist = UrlAllowlist()
		fetch_fn = self.fetch_factory(allowlist)
		cache_enabled = self.settings.USE_CACHE and not self.settings.FAULT_INJECT.strip()
		fetch_fn = _cached_fetch_fn(
			fetch_fn,
			DiskCache(self.settings.CACHE_DIR, enabled=cache_enabled),
			tracer,
			budget,
		)
		for provider in self.search_providers:
			set_run_context = getattr(provider, "set_run_context", None)
			if callable(set_run_context):
				set_run_context(tracer, budget)
		if self.settings.FAULT_INJECT.strip():
			injector = FaultInjector(
				self.settings.FAULT_INJECT,
				seed=int(run_id[-6:], 16),
			)
			fetch_fn = wrap_fetch(fetch_fn, injector)
		return RunContext(
			run_id=run_id,
			budget=budget,
			tracer=tracer,
			allowlist=allowlist,
			fetch_fn=fetch_fn,
		)


def build_default_deps(settings: Settings | None = None) -> Deps:
	"""Build configured Groq, search-provider, and fetch dependencies."""
	active_settings = settings or load_settings()
	if active_settings.LLM_API_KEY is None or not active_settings.LLM_API_KEY.get_secret_value():
		raise ValueError("LLM_API_KEY is required to build default dependencies")

	provider_names = [
		provider.strip().lower()
		for provider in active_settings.SEARCH_PROVIDERS.split(",")
		if provider.strip()
	]
	providers: list[SearchProvider] = []
	cache_enabled = active_settings.USE_CACHE and not active_settings.FAULT_INJECT.strip()
	cache = DiskCache(active_settings.CACHE_DIR, enabled=cache_enabled)
	for provider_name in provider_names:
		if provider_name == "tavily":
			if active_settings.TAVILY_API_KEY is None or not (
				active_settings.TAVILY_API_KEY.get_secret_value()
			):
				raise ValueError(
					"TAVILY_API_KEY is required because tavily is in SEARCH_PROVIDERS"
				)
			provider = TavilySearchProvider(active_settings)
			providers.append(CachedSearchProvider(provider, cache))
		elif provider_name == "ddgs":
			provider = DDGSSearchProvider(active_settings)
			providers.append(CachedSearchProvider(provider, cache))
		else:
			raise ValueError(f"unsupported search provider: {provider_name}")

	fetch_factory: FetchFactory = partial(
		_make_fetch_fn,
		settings=active_settings,
	)
	return Deps(
		settings=active_settings,
		llm=GroqClient(active_settings),
		search_providers=providers,
		fetch_factory=fetch_factory,
	)


def _make_fetch_fn(allowlist: UrlAllowlist, *, settings: Settings) -> FetchFn:
	return partial(
		fetch_page,
		allowlist=allowlist,
		settings=settings,
	)


def _cached_fetch_fn(
	fetch_fn: FetchFn,
	cache: DiskCache,
	tracer: Tracer,
	budget: Budget,
) -> FetchFn:
	def cached_fetch(url: str) -> FetchResult:
		normalized_url = normalize_url(url)
		inputs = {"url": normalized_url}
		found, value = cache.load("fetch", inputs)
		if found:
			tracer.event("note", "fetch", cache_hit=True)
			if budget.fetches > 0:
				budget.fetches -= 1
			return FetchResult.model_validate(value)
		result = fetch_fn(url)
		if result.status is FetchStatus.OK:
			cache.store("fetch", inputs, result.model_dump(mode="json"))
		return result

	return cached_fetch
