"""Tavily-backed SearchProvider implementation."""

from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

import requests
from tavily import TavilyClient
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    KeylessUnsupportedEndpointError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)
from tavily.errors import TimeoutError as TavilyTimeoutError

from research_agent.config import Settings
from research_agent.errors import ToolFailure
from research_agent.schemas import ErrorClass, SearchArgs, SearchResult, ToolError
from research_agent.tools.base import SearchProvider


def _failure(error_class: ErrorClass, message: str, status: int | None = None) -> ToolFailure:
    return ToolFailure(
        ToolError(tool="tavily", error_class=error_class, message=message, http_status=status)
    )


def _execute_with_timeout(fn: Any, timeout_s: float) -> Any:  # noqa: ANN401
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeoutError as err:
        future.cancel()
        raise _failure(ErrorClass.TRANSIENT, "Tavily search timed out") from err
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _http_status(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _map_error(error: Exception) -> ToolFailure:
    if isinstance(error, UsageLimitExceededError):
        detail = str(error).lower()
        if "rate limit" in detail or "too many requests" in detail:
            return _failure(ErrorClass.TRANSIENT, "Tavily rate limit reached", 429)
        return _failure(ErrorClass.CLIENT, "Tavily usage quota exceeded")
    if isinstance(
        error,
        (
            InvalidAPIKeyError,
            BadRequestError,
            ForbiddenError,
            MissingAPIKeyError,
            KeylessUnsupportedEndpointError,
        ),
    ):
        return _failure(
            ErrorClass.CLIENT,
            f"Tavily rejected the request ({type(error).__name__})",
        )
    if isinstance(
        error,
        (TavilyTimeoutError, requests.Timeout, requests.ConnectionError, socket.timeout),
    ):
        return _failure(ErrorClass.TRANSIENT, f"Tavily transport failed ({type(error).__name__})")
    status = _http_status(error)
    if status == 429 or (status is not None and status >= 500):
        return _failure(ErrorClass.TRANSIENT, "Tavily returned a retriable HTTP status", status)
    if status is not None and 400 <= status < 500:
        return _failure(ErrorClass.CLIENT, "Tavily rejected the request", status)
    if isinstance(error, requests.RequestException):
        return _failure(ErrorClass.TRANSIENT, f"Tavily request failed ({type(error).__name__})")
    return _failure(ErrorClass.UNKNOWN, f"Tavily SDK failed ({type(error).__name__})")


def _normalize_item(item: object) -> SearchResult | None:
    if not isinstance(item, dict):
        return None
    url = item.get("url")
    title = item.get("title")
    snippet = item.get("content", item.get("snippet"))
    if (
        not isinstance(url, str)
        or not url
        or not isinstance(title, str)
        or not isinstance(snippet, str)
    ):
        return None
    score = item.get("score")
    published_date = item.get("published_date")
    return SearchResult(
        url=url,
        title=title,
        snippet=snippet,
        score=score if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
        published_date=published_date if isinstance(published_date, str) else None,
        provider="tavily",
    )


class TavilySearchProvider(SearchProvider):
    """Normalize official Tavily search responses to ``SearchResult`` models."""

    name = "tavily"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:  # noqa: ANN401
        self.settings = settings
        api_key = (
            settings.TAVILY_API_KEY.get_secret_value()
            if settings.TAVILY_API_KEY is not None
            else None
        )
        self.client = client if client is not None else TavilyClient(api_key=api_key)
        self.last_dropped_count = 0

    def search(self, args: SearchArgs) -> list[SearchResult]:
        self.last_dropped_count = 0
        try:
            response = _execute_with_timeout(
                lambda: self.client.search(
                    query=args.query,
                    search_depth=args.depth,
                    max_results=args.max_results,
                    timeout=self.settings.SEARCH_TIMEOUT_S,
                ),
                self.settings.SEARCH_TIMEOUT_S,
            )
        except ToolFailure:
            raise
        except Exception as err:
            raise _map_error(err) from err
        if not isinstance(response, dict) or not isinstance(response.get("results"), list):
            raise _failure(ErrorClass.CONTENT, "Tavily response has no results list")
        results: list[SearchResult] = []
        for item in response["results"]:
            normalized = _normalize_item(item)
            if normalized is None:
                self.last_dropped_count += 1
            else:
                results.append(normalized)
        return results