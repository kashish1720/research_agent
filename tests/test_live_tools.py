"""Opt-in live smoke tests; normal pytest excludes the ``live`` marker."""

from __future__ import annotations

import os

import pytest

from research_agent.config import Settings
from research_agent.schemas import SearchArgs
from research_agent.tools.ddgs_search import DDGSSearchProvider
from research_agent.tools.fetch_page import fetch_page
from research_agent.tools.tavily_search import TavilySearchProvider
from research_agent.tools.url_safety import UrlAllowlist


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("TAVILY_API_KEY"), reason="TAVILY_API_KEY is not configured")
def test_live_tavily_search() -> None:
    provider = TavilySearchProvider(Settings())
    assert provider.search(SearchArgs(query="Python programming language", max_results=1))


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_DDGS"),
    reason="Set RUN_LIVE_DDGS=1 to enable live DDGS",
)
def test_live_ddgs_search() -> None:
    provider = DDGSSearchProvider(Settings())
    assert provider.search(SearchArgs(query="Python programming language", max_results=1))


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("LIVE_FETCH_URL"),
    reason="Set LIVE_FETCH_URL to enable live fetch",
)
def test_live_fetch() -> None:
    url = os.environ["LIVE_FETCH_URL"]
    allowlist = UrlAllowlist()
    allowlist.add([url])
    result = fetch_page(url, allowlist=allowlist, settings=Settings())
    assert result.status.value == "OK"
    assert result.text