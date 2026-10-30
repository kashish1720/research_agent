"""Tests for default dependency and per-run construction."""

from __future__ import annotations

import re

import pytest
from pydantic import SecretStr

from research_agent.config import Settings
from research_agent.deps import Deps, build_default_deps
from research_agent.llm import FakeLLMClient
from research_agent.schemas import FetchStatus
from research_agent.tools.url_safety import UrlAllowlist


def test_default_deps_reports_missing_llm_key() -> None:
    with pytest.raises(ValueError, match="LLM_API_KEY is required"):
        build_default_deps(Settings(SEARCH_PROVIDERS="ddgs"))


def test_default_deps_reports_missing_tavily_key() -> None:
    settings = Settings(
        LLM_API_KEY=SecretStr("test-key"),
        SEARCH_PROVIDERS="tavily",
    )

    with pytest.raises(ValueError, match="TAVILY_API_KEY is required"):
        build_default_deps(settings)


def test_default_deps_builds_selected_provider_and_models() -> None:
    settings = Settings(
        LLM_API_KEY=SecretStr("test-key"),
        LLM_MODEL_CHEAP="openai/gpt-oss-20b",
        LLM_MODEL_STRONG="openai/gpt-oss-120b",
        SEARCH_PROVIDERS="ddgs",
    )

    deps = build_default_deps(settings)

    assert [provider.name for provider in deps.search_providers] == ["ddgs"]
    assert deps.llm.model_for("cheap") == "openai/gpt-oss-20b"  # type: ignore[attr-defined]
    assert deps.llm.model_for("strong") == "openai/gpt-oss-120b"  # type: ignore[attr-defined]


def test_new_run_creates_context_and_fault_wrapped_fetch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(LOG_DIR=str(tmp_path), FAULT_INJECT="empty:1")
    deps = Deps(
        settings=settings,
        llm=FakeLLMClient(),
        search_providers=[],
        fetch_factory=lambda allowlist: lambda url: _fetch_stub(allowlist, url),
    )

    run = deps.new_run("test question")
    result = run.fetch_fn("https://example.com")
    run.tracer.event("note", "test", details="run initialized")

    assert re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{6}", run.run_id)
    assert run.budget.elapsed() >= 0
    assert run.tracer.path.exists()
    assert result.status is FetchStatus.EMPTY


def test_new_run_uses_fresh_allowlist() -> None:
    settings = Settings(FAULT_INJECT="")
    deps = Deps(
        settings=settings,
        llm=FakeLLMClient(),
        search_providers=[],
        fetch_factory=lambda allowlist: lambda url: _fetch_stub(allowlist, url),
    )

    first = deps.new_run("first")
    second = deps.new_run("second")
    first.allowlist.add(["https://example.com"])

    assert first.allowlist.contains("https://example.com")
    assert not second.allowlist.contains("https://example.com")


def _fetch_stub(allowlist: UrlAllowlist, url: str):  # type: ignore[no-untyped-def]
    from research_agent.schemas import FetchResult

    return FetchResult(
        url=url,
        status=FetchStatus.OK if allowlist.contains(url) else FetchStatus.SECURITY_BLOCKED,
    )