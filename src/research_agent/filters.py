"""Content filtering, source scoring, URL normalization, and injection scanning."""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Iterable
from datetime import date
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import tldextract

from research_agent.config import Settings
from research_agent.schemas import SearchResult

_EXTRACT_DOMAIN = tldextract.TLDExtract(suffix_list_urls=())
_TRACKING_PARAMS = {
    "gclid",
    "fbclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "igshid",
}
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "how", "in", "is",
    "of", "on", "or", "the", "to", "with",
}
_TOKENS_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_LISTICLE_RE = re.compile(r"\b(?:top|best)[\s-]*(?:\d+|n)\b", re.IGNORECASE)
_PROMOTIONAL_RE = re.compile(
    r"\b(?:sponsored|affiliate|buy now|purchase now|subscribe|request a demo|"
    r"our platform|limited.time offer|discount code)\b",
    re.IGNORECASE,
)
_INJECTION_PATTERNS = (
    (re.compile(r"ignore\s+(?:the\s+)?(?:previous|above)\s+instructions?", re.I),
     "ignore previous/above instructions"),
    (re.compile(r"disregard\s+the\s+above", re.I), "disregard the above"),
    (re.compile(r"you\s+are\s+now\b", re.I), "you are now"),
    (re.compile(r"new\s+instructions\s*:", re.I), "new instructions:"),
    (re.compile(r"reveal\s+your\s+system\s+prompt", re.I), "reveal your system prompt"),
    (re.compile(r"do\s+not\s+tell\s+the\s+user", re.I), "do not tell the user"),
    (re.compile(r"\bsystem\s*:", re.I), "system:"),
    (re.compile(r"\bassistant\s*:", re.I), "assistant:"),
)
_DOMAIN_SCORES = {
    "arxiv.org": 0.9,
    "github.com": 0.9,
    "huggingface.co": 0.9,
    "readthedocs.io": 0.9,
    "wikipedia.org": 0.9,
    "reddit.com": 0.4,
    "quora.com": 0.4,
    "pinterest.com": 0.2,
    "facebook.com": 0.2,
    "instagram.com": 0.2,
    "tiktok.com": 0.2,
}
_MULTI_LABEL_SUFFIXES = {"co.uk", "com.au", "co.in", "com.br", "co.jp"}


def normalize_url(url: str) -> str:
    """Canonicalize URL case, tracking parameters, fragments, and trailing slashes."""
    try:
        parts = urlsplit(url.strip())
        scheme = parts.scheme.lower()
        hostname = parts.hostname
        if not hostname:
            return url.strip()
        host = hostname.lower()
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        userinfo = ""
        if "@" in parts.netloc:
            userinfo = parts.netloc.rsplit("@", 1)[0] + "@"
        netloc = userinfo + host
        if parts.port is not None:
            netloc += f":{parts.port}"
        path = parts.path or "/"
        if path != "/":
            path = path.rstrip("/") or "/"
        params = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.casefold().startswith("utm_") and key.casefold() not in _TRACKING_PARAMS
        ]
        params.sort(key=lambda pair: (pair[0].casefold(), pair[0], pair[1]))
        return urlunsplit((scheme, netloc, path, urlencode(params, doseq=True), ""))
    except (UnicodeError, ValueError):
        return url.strip()


def registered_domain(url: str) -> str:
    """Return the registered domain using only tldextract's bundled suffix snapshot."""
    try:
        parts = urlsplit(url if "://" in url else f"//{url}")
        hostname = (parts.hostname or "").rstrip(".").encode("idna").decode("ascii").lower()
        if not hostname:
            return url
        extracted = _EXTRACT_DOMAIN(hostname)
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}".lower()
        return _fallback_domain(hostname)
    except (AttributeError, UnicodeError, ValueError):
        return _fallback_domain(url)


def _fallback_domain(hostname: str) -> str:
    labels = hostname.strip(".").lower().split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in _MULTI_LABEL_SUFFIXES:
        return ".".join(labels[-3:])
    if len(labels) >= 2:
        return ".".join(labels[-2:])
    return hostname.strip(".").lower()


def dedupe_results(
    results: Iterable[SearchResult],
    score_fn: Callable[[SearchResult], float] | None = None,
) -> list[SearchResult]:
    """Keep the highest-scored URL and near-duplicate title per registered domain."""
    retained: list[SearchResult] = []
    scores: list[float] = []
    for result in results:
        result_score = (
            score_fn(result)
            if score_fn is not None
            else (result.score if result.score is not None else 0.0)
        )
        result_url = normalize_url(result.url)
        result_domain = registered_domain(result.url)
        duplicate_index: int | None = None
        for index, previous in enumerate(retained):
            same_url = result_url == normalize_url(previous.url)
            same_domain = result_domain == registered_domain(previous.url)
            similar_title = difflib.SequenceMatcher(
                None,
                result.title.casefold(),
                previous.title.casefold(),
            ).ratio() >= 0.9
            if same_url or (same_domain and similar_title):
                duplicate_index = index
                break
        if duplicate_index is None:
            retained.append(result)
            scores.append(result_score)
        elif result_score > scores[duplicate_index]:
            retained[duplicate_index] = result
            scores[duplicate_index] = result_score
    return retained


def domain_score(url: str) -> float:
    """Score common high- and low-trust host classes, defaulting to neutral."""
    try:
        hostname = (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return 0.5
    if hostname.endswith((".gov", ".edu")):
        return 0.9
    if hostname.startswith(("docs.", "developer.")):
        return 0.9
    for domain, score in _DOMAIN_SCORES.items():
        if hostname == domain or hostname.endswith(f".{domain}"):
            return score
    return 0.5


def flags_for(url: str, title: str, snippet: str) -> list[str]:
    """Identify listicle and promotional source characteristics."""
    content = f"{url} {title} {snippet}"
    flags: list[str] = []
    if _LISTICLE_RE.search(content):
        flags.append("listicle")
    if _PROMOTIONAL_RE.search(content):
        flags.append("promotional")
    return flags


def _content_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKENS_RE.findall(text)
        if token.casefold() not in _STOPWORDS
    }


def relevance(
    sub_question_text: str,
    queries: Iterable[str],
    title: str,
    snippet: str,
    provider_score: float | None = None,
) -> float:
    """Combine provider relevance (when present) with query/content token overlap."""
    query_tokens = _content_tokens(" ".join((sub_question_text, *queries)))
    content_tokens = _content_tokens(f"{title} {snippet}")
    overlap = (
        len(query_tokens & content_tokens) / len(query_tokens)
        if query_tokens
        else 0.0
    )
    if provider_score is None:
        return overlap
    bounded_provider_score = min(1.0, max(0.0, provider_score))
    return 0.6 * bounded_provider_score + 0.4 * overlap


def recency_factor(
    published_date: str | None,
    time_sensitive: bool,
    today: date,
    stale_after_days: int,
) -> tuple[float, list[str]]:
    """Penalize missing or stale dates only when a question is time-sensitive."""
    if not time_sensitive:
        return 1.0, []
    if not published_date:
        return 0.8, []
    try:
        published = date.fromisoformat(published_date[:10])
    except ValueError:
        return 0.8, []
    if (today - published).days > stale_after_days:
        return 0.4, ["stale"]
    return 1.0, []


def final_score(
    relevance_score: float,
    source_domain_score: float,
    recency: float,
    flags: Iterable[str],
    settings: Settings,
) -> float:
    """Apply configured relevance/domain/recency weights and the listicle penalty."""
    score = (
        settings.W_RELEVANCE * relevance_score
        + settings.W_DOMAIN * source_domain_score
        + settings.W_RECENCY * recency
    )
    if "listicle" in flags:
        score -= 0.15
    return min(1.0, max(0.0, score))


def score_result(
    result: SearchResult,
    *,
    sub_question_text: str,
    queries: Iterable[str],
    time_sensitive: bool,
    today: date,
    settings: Settings,
) -> tuple[float, list[str]]:
    """Return a source's final score and diagnostic flags."""
    flags = flags_for(result.url, result.title, result.snippet)
    relevance_score = relevance(
        sub_question_text,
        queries,
        result.title,
        result.snippet,
        result.score,
    )
    recency, recency_flags = recency_factor(
        result.published_date,
        time_sensitive,
        today,
        settings.STALE_AFTER_DAYS,
    )
    flags.extend(recency_flags)
    score = final_score(
        relevance_score,
        domain_score(result.url),
        recency,
        flags,
        settings,
    )
    return score, flags


def scan_for_injection(text: str) -> tuple[str, list[str]]:
    """Remove instruction-like lines from untrusted source text."""
    clean_lines: list[str] = []
    matched_names: list[str] = []
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        names = [
            name
            for pattern, name in _INJECTION_PATTERNS
            if pattern.search(content)
        ]
        if names:
            matched_names.extend(names)
        else:
            clean_lines.append(line)
    return "".join(clean_lines), matched_names
