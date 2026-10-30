"""DDGS-backed SearchProvider implementation."""

from __future__ import annotations

import re
import socket
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

import httpx
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from research_agent.config import Settings
from research_agent.errors import ToolFailure
from research_agent.schemas import ErrorClass, SearchArgs, SearchResult, ToolError
from research_agent.tools.base import SearchProvider


def _failure(error_class: ErrorClass, message: str, status: int | None = None) -> ToolFailure:
    return ToolFailure(
        ToolError(tool="ddgs", error_class=error_class, message=message, http_status=status)
    )


def _execute_with_timeout(fn: Any, timeout_s: float) -> Any:  # noqa: ANN401
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeoutError as err:
        future.cancel()
        raise _failure(ErrorClass.TRANSIENT, "DDGS search timed out") from err
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _http_status(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    match = re.search(r"\bHTTP\s+(\d{3})\b", str(error), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _map_error(error: Exception) -> ToolFailure:
    if isinstance(
        error,
        (TimeoutException, RatelimitException, httpx.TimeoutException, socket.timeout),
    ):
        return _failure(ErrorClass.TRANSIENT, f"DDGS request failed ({type(error).__name__})")
    status = _http_status(error)
    if status == 429 or (status is not None and status >= 500):
        return _failure(ErrorClass.TRANSIENT, "DDGS returned a retriable HTTP status", status)
    if status is not None and 400 <= status < 500:
        return _failure(ErrorClass.CLIENT, "DDGS rejected the request", status)
    detail = str(error).lower()
    client_error_terms = ("invalid api key", "unauthorized", "quota", "plan limit", "bad request")
    if any(term in detail for term in client_error_terms):
        return _failure(ErrorClass.CLIENT, f"DDGS rejected the request ({type(error).__name__})")
    if isinstance(error, (httpx.NetworkError, DDGSException, ConnectionError)):
        return _failure(ErrorClass.TRANSIENT, f"DDGS transport failed ({type(error).__name__})")
    return _failure(ErrorClass.UNKNOWN, f"DDGS SDK failed ({type(error).__name__})")


def _normalize_item(item: object) -> SearchResult | None:
    if not isinstance(item, dict):
        return None
    url = item.get("href", item.get("url"))
    title = item.get("title")
    snippet = item.get("body", item.get("snippet", item.get("content")))
    if (
        not isinstance(url, str)
        or not url
        or not isinstance(title, str)
        or not isinstance(snippet, str)
    ):
        return None
    score = item.get("score")
    published_date = item.get("published", item.get("published_date", item.get("date")))
    return SearchResult(
        url=url,
        title=title,
        snippet=snippet,
        score=score if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
        published_date=published_date if isinstance(published_date, str) else None,
        provider="ddgs",
    )


class DDGSSearchProvider(SearchProvider):
    """Normalize official DDGS text-search responses to ``SearchResult`` models."""

    name = "ddgs"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:  # noqa: ANN401
        self.settings = settings
        self.client = client if client is not None else DDGS(timeout=settings.SEARCH_TIMEOUT_S)
        self.last_dropped_count = 0

    def search(self, args: SearchArgs) -> list[SearchResult]:
        self.last_dropped_count = 0
        try:
            response = _execute_with_timeout(
                lambda: self.client.text(args.query, max_results=args.max_results),
                self.settings.SEARCH_TIMEOUT_S,
            )
        except ToolFailure:
            raise
        except Exception as err:
            raise _map_error(err) from err
        if isinstance(response, dict):
            if not isinstance(response.get("results"), list):
                raise _failure(ErrorClass.CONTENT, "DDGS response has no results list")
            items = response["results"]
        elif isinstance(response, list):
            items = response
        else:
            raise _failure(ErrorClass.CONTENT, "DDGS response has no results list")
        results: list[SearchResult] = []
        for item in items:
            normalized = _normalize_item(item)
            if normalized is None:
                self.last_dropped_count += 1
            else:
                results.append(normalized)
        return results