"""Secure, bounded HTTP fetching and text extraction."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from urllib.parse import urljoin

import httpx

from research_agent.config import Settings
from research_agent.errors import ToolFailure
from research_agent.schemas import ErrorClass, FetchResult, FetchStatus, ToolError
from research_agent.tools.url_safety import Resolver, UrlAllowlist, validate_url

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_ALLOWED_CONTENT_TYPES = {"text/html", "application/xhtml+xml", "text/plain"}


def _tool_failure(error_class: ErrorClass, message: str, status: int | None = None) -> ToolFailure:
    return ToolFailure(
        ToolError(tool="fetch", error_class=error_class, message=message, http_status=status)
    )


def _extract_html(html: str, url: str) -> tuple[str, str | None, str | None]:
    """Use Trafilatura's verified metadata extractor lazily."""
    from trafilatura import extract_with_metadata

    document = extract_with_metadata(html, url=url)
    if document is None:
        return "", None, None
    return (
        getattr(document, "text", None) or "",
        getattr(document, "title", None),
        getattr(document, "date", None),
    )


def _iter_chunks(response: httpx.Response) -> Iterator[bytes]:
    yield from response.iter_bytes(chunk_size=65_536)


def _security_result(url: str, final_url: str | None = None) -> FetchResult:
    return FetchResult(url=url, final_url=final_url, status=FetchStatus.SECURITY_BLOCKED)


def fetch_page(
    url: str,
    *,
    allowlist: UrlAllowlist,
    settings: Settings,
    client: httpx.Client | None = None,
    resolver: Resolver | None = None,
) -> FetchResult:
    """Fetch an allowlisted URL while validating every redirect hop."""
    if not allowlist.contains(url):
        return _security_result(url)

    active_resolver = resolver if resolver is not None else socket.getaddrinfo
    current_url = url
    timeout = httpx.Timeout(settings.FETCH_TIMEOUT_S)
    owns_client = client is None
    active_client = client or httpx.Client(timeout=timeout, follow_redirects=False)
    redirect_count = 0
    try:
        while True:
            try:
                validate_url(current_url, resolver=active_resolver)
            except ToolFailure as failure:
                if failure.error.error_class is ErrorClass.SECURITY:
                    return _security_result(url, current_url)
                raise

            try:
                with active_client.stream(
                    "GET",
                    current_url,
                    headers={"User-Agent": settings.USER_AGENT},
                    timeout=timeout,
                    follow_redirects=False,
                ) as response:
                    status = response.status_code
                    if status in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location:
                            return FetchResult(
                                url=url,
                                final_url=current_url,
                                status=FetchStatus.HTTP_ERROR,
                                http_status=status,
                            )
                        if redirect_count >= settings.MAX_REDIRECTS:
                            return FetchResult(
                                url=url,
                                final_url=current_url,
                                status=FetchStatus.HTTP_ERROR,
                                http_status=status,
                            )
                        current_url = urljoin(current_url, location)
                        redirect_count += 1
                        continue
                    if status in {401, 403, 451}:
                        return FetchResult(
                            url=url,
                            final_url=current_url,
                            status=FetchStatus.BLOCKED,
                            http_status=status,
                        )
                    if status in {404, 410}:
                        return FetchResult(
                            url=url,
                            final_url=current_url,
                            status=FetchStatus.NOT_FOUND,
                            http_status=status,
                        )
                    if status == 429 or status >= 500:
                        raise _tool_failure(
                            ErrorClass.TRANSIENT,
                            "fetch provider returned a retriable HTTP status",
                            status,
                        )
                    if 400 <= status < 500:
                        return FetchResult(
                            url=url,
                            final_url=current_url,
                            status=FetchStatus.HTTP_ERROR,
                            http_status=status,
                        )

                    content_type = response.headers.get("content-type", "")
                    media_type = content_type.split(";", 1)[0].strip().lower()
                    if media_type not in _ALLOWED_CONTENT_TYPES:
                        return FetchResult(
                            url=url,
                            final_url=current_url,
                            status=FetchStatus.NON_HTML,
                            http_status=status,
                        )

                    content_length = response.headers.get("content-length")
                    if content_length is not None:
                        try:
                            if int(content_length) > settings.MAX_FETCH_BYTES:
                                return FetchResult(
                                    url=url,
                                    final_url=current_url,
                                    status=FetchStatus.TOO_LARGE,
                                    http_status=status,
                                )
                        except ValueError:
                            pass

                    body = bytearray()
                    for chunk in _iter_chunks(response):
                        if len(body) + len(chunk) > settings.MAX_FETCH_BYTES:
                            return FetchResult(
                                url=url,
                                final_url=current_url,
                                status=FetchStatus.TOO_LARGE,
                                http_status=status,
                            )
                        body.extend(chunk)
            except httpx.RequestError as err:
                raise _tool_failure(
                    ErrorClass.TRANSIENT,
                    f"fetch transport failed ({type(err).__name__})",
                ) from err

            html = bytes(body).decode("utf-8", errors="replace")
            try:
                text, title, published_date = _extract_html(html, current_url)
            except Exception as err:
                return FetchResult(
                    url=url,
                    final_url=current_url,
                    status=FetchStatus.EMPTY,
                    http_status=status,
                    error=ToolError(
                        tool="fetch",
                        error_class=ErrorClass.CONTENT,
                        message=f"content extraction failed ({type(err).__name__})",
                        http_status=status,
                    ),
                )
            if len(text) < settings.MIN_PAGE_TEXT_CHARS:
                return FetchResult(
                    url=url,
                    final_url=current_url,
                    status=FetchStatus.EMPTY,
                    text=text,
                    title=title,
                    published_date=published_date,
                    http_status=status,
                )
            truncated = len(text) > settings.MAX_FETCH_TEXT_CHARS
            if truncated:
                text = text[: settings.MAX_FETCH_TEXT_CHARS]
            return FetchResult(
                url=url,
                final_url=current_url,
                status=FetchStatus.OK,
                text=text,
                title=title,
                published_date=published_date,
                http_status=status,
                truncated=truncated,
            )
    finally:
        if owns_client:
            active_client.close()