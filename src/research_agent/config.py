"""Configuration settings loaded from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator, model_validator


class Settings(BaseModel):
    """All runtime settings for the research agent.

    Env var name is the UPPERCASE field name.
    Secrets are never revealed in repr, str, or model_dump.
    """

    model_config = ConfigDict(
        # Do NOT expose SecretStr values in any default serialization
        populate_by_name=True,
    )

    # ── Budgets ───────────────────────────────────────────────────────────────
    MAX_SEARCHES: int = 8
    MAX_FETCHES: int = 12
    MAX_LLM_CALLS: int = 30
    MAX_TOTAL_TOKENS: int = 200_000
    MAX_REFINE_LOOPS: int = 1
    WALL_TIME_S: int = 120
    MAX_WORKERS: int = 4

    # ── Timeouts / retries ────────────────────────────────────────────────────
    SEARCH_TIMEOUT_S: int = 10
    FETCH_TIMEOUT_S: int = 12
    LLM_TIMEOUT_S: int = 60
    MAX_RETRIES: int = 2
    BACKOFF_BASE_S: float = 1.0
    MAX_CORRECTIONS: int = 2
    MAX_REDIRECTS: int = 3

    # ── Fetch ─────────────────────────────────────────────────────────────────
    MAX_FETCH_BYTES: int = 2_000_000
    MAX_FETCH_TEXT_CHARS: int = 60_000
    MIN_PAGE_TEXT_CHARS: int = 200
    USER_AGENT: str = "ResearchAgentBot/0.1 (educational project)"

    # ── Planning ──────────────────────────────────────────────────────────────
    MAX_SUBQUESTIONS: int = 5
    QUERIES_PER_SUBQUESTION: int = 2

    # ── Search ────────────────────────────────────────────────────────────────
    SEARCH_RESULTS_PER_QUERY: int = 5
    SEARCH_PROVIDERS: str = "tavily,ddgs"

    # ── Selection ─────────────────────────────────────────────────────────────
    TOP_K_FETCH_PER_SUBQ: int = 2
    SNIPPET_ONLY_TOP_K: int = 3
    MIN_SOURCE_SCORE: float = 0.35
    W_RELEVANCE: float = 0.5
    W_DOMAIN: float = 0.35
    W_RECENCY: float = 0.15
    STALE_AFTER_DAYS: int = 730
    MIN_SNIPPET_CHARS: int = 80
    INJECTION_DROP_THRESHOLD: int = 3

    # ── Extraction ────────────────────────────────────────────────────────────
    EXTRACT_CHUNK_CHARS: int = 1200
    EXTRACT_MAX_CHARS_PER_SOURCE: int = 6000
    MAX_EVIDENCE_PER_SOURCE: int = 3
    PASSAGE_MIN_WORDS: int = 5
    PASSAGE_MAX_WORDS: int = 40

    # ── Synthesis ─────────────────────────────────────────────────────────────
    MAX_EVIDENCE_FOR_SYNTH: int = 30
    STRICT_NUMERIC_CHECK: bool = True

    # ── Sufficiency ───────────────────────────────────────────────────────────
    MIN_COVERAGE_TO_ANSWER: float = 0.5
    TARGET_COVERAGE: float = 0.8

    # ── Confidence ────────────────────────────────────────────────────────────
    HIGH_MIN_AVG_QUALITY: float = 0.65
    HIGH_MAX_SNIPPET_RATIO: float = 0.25
    HIGH_MIN_DOMAINS: int = 2
    MEDIUM_MIN_COVERAGE: float = 0.7
    MEDIUM_MIN_AVG_QUALITY: float = 0.5
    MEDIUM_MAX_CONFLICTS: int = 1

    # ── Secrets / providers ───────────────────────────────────────────────────
    TAVILY_API_KEY: SecretStr | None = None
    LLM_API_KEY: SecretStr | None = None
    LLM_PROVIDER: str | None = None
    LLM_MODEL_CHEAP: str | None = None
    LLM_MODEL_STRONG: str | None = None

    # ── Optional price floats ─────────────────────────────────────────────────
    PRICE_IN_PER_MTOK_CHEAP: float | None = None
    PRICE_OUT_PER_MTOK_CHEAP: float | None = None
    PRICE_IN_PER_MTOK_STRONG: float | None = None
    PRICE_OUT_PER_MTOK_STRONG: float | None = None

    # ── Paths / faults ────────────────────────────────────────────────────────
    LOG_DIR: str = "logs"
    RUNS_DIR: str = "runs"
    CACHE_DIR: str = "cache"
    USE_CACHE: bool = True
    FAULT_INJECT: str = ""

    # ──────────────────────────────────────────────────────────────────────────
    # Cross-field validators
    # ──────────────────────────────────────────────────────────────────────────

    @field_validator(
        "MAX_SEARCHES",
        "MAX_FETCHES",
        "MAX_LLM_CALLS",
        "MAX_TOTAL_TOKENS",
        "MAX_REFINE_LOOPS",
        "WALL_TIME_S",
        "MAX_WORKERS",
        "SEARCH_TIMEOUT_S",
        "FETCH_TIMEOUT_S",
        "LLM_TIMEOUT_S",
        "MAX_RETRIES",
        "MAX_CORRECTIONS",
        "MAX_REDIRECTS",
        "MAX_FETCH_BYTES",
        "MAX_FETCH_TEXT_CHARS",
        "MIN_PAGE_TEXT_CHARS",
        "MAX_SUBQUESTIONS",
        "QUERIES_PER_SUBQUESTION",
        "SEARCH_RESULTS_PER_QUERY",
        "TOP_K_FETCH_PER_SUBQ",
        "SNIPPET_ONLY_TOP_K",
        "STALE_AFTER_DAYS",
        "MIN_SNIPPET_CHARS",
        "INJECTION_DROP_THRESHOLD",
        "EXTRACT_CHUNK_CHARS",
        "EXTRACT_MAX_CHARS_PER_SOURCE",
        "MAX_EVIDENCE_PER_SOURCE",
        "PASSAGE_MIN_WORDS",
        "PASSAGE_MAX_WORDS",
        "MAX_EVIDENCE_FOR_SYNTH",
        "HIGH_MIN_DOMAINS",
        "MEDIUM_MAX_CONFLICTS",
        mode="after",
    )
    @classmethod
    def _validate_budget_gte_1(cls, v: int) -> int:
        if v < 1:
            raise ValueError("budget/count fields must be >= 1")
        return v

    @field_validator("BACKOFF_BASE_S", mode="after")
    @classmethod
    def _validate_backoff(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("BACKOFF_BASE_S must be > 0")
        return v

    @model_validator(mode="after")
    def _cross_field_checks(self) -> Settings:
        # Weight sum
        w_sum = self.W_RELEVANCE + self.W_DOMAIN + self.W_RECENCY
        if abs(w_sum - 1.0) > 1e-6:
            raise ValueError(
                f"W_RELEVANCE + W_DOMAIN + W_RECENCY must equal 1.0, got {w_sum}"
            )
        # Coverage ordering
        if self.MIN_COVERAGE_TO_ANSWER > self.TARGET_COVERAGE:
            raise ValueError(
                "MIN_COVERAGE_TO_ANSWER must be <= TARGET_COVERAGE"
            )
        return self

    # ──────────────────────────────────────────────────────────────────────────
    # Public helpers
    # ──────────────────────────────────────────────────────────────────────────

    def redacted_dict(self) -> dict[str, Any]:
        """Return all settings with secret values replaced by '***' (or None)."""
        result: dict[str, Any] = {}
        for name in type(self).model_fields:
            raw = getattr(self, name)
            # Detect by runtime type: if the value is a SecretStr, redact it
            if isinstance(raw, SecretStr):
                result[name] = "***"
            else:
                result[name] = raw
        return result

# ──────────────────────────────────────────────────────────────────────────────
# Type-cast helpers
# ──────────────────────────────────────────────────────────────────────────────

_BOOL_TRUE = {"true", "1", "yes"}
_BOOL_FALSE = {"false", "0", "no"}


def _cast(field_name: str, raw: str, python_type: Any) -> Any:  # noqa: ANN401
    """Cast *raw* string to *python_type*, raising ValueError on failure."""
    # Unwrap Optional / Union to find the core type
    core = _unwrap_optional(python_type)

    if core is bool:
        lower = raw.strip().lower()
        if lower in _BOOL_TRUE:
            return True
        if lower in _BOOL_FALSE:
            return False
        raise ValueError(
            f"Invalid value for {field_name!r}: {raw!r}. Expected true/false/1/0."
        )

    if core is int:
        try:
            return int(raw)
        except ValueError as err:
            raise ValueError(f"Invalid integer for {field_name!r}: {raw!r}") from err

    if core is float:
        try:
            return float(raw)
        except ValueError as err:
            raise ValueError(f"Invalid float for {field_name!r}: {raw!r}") from err

    if core is SecretStr:
        return SecretStr(raw)

    # str and any other type — return as-is
    return raw


def _unwrap_optional(annotation: Any) -> Any:
    """Return the first non-None type arg from Optional / Union, else annotation."""
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        args = [a for a in getattr(annotation, "__args__", ()) if a is not type(None)]
        if args:
            return args[0]
    return annotation


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build a Settings instance.

    If *env* is None, python-dotenv's load_dotenv() is called first, then
    os.environ is used.  Pass an explicit mapping for tests.
    """
    if env is None:
        load_dotenv()
        env = os.environ  # type: ignore[assignment]

    overrides: dict[str, Any] = {}
    model_fields = Settings.model_fields

    for field_name, field_info in model_fields.items():
        env_key = field_name.upper()  # already uppercase by convention
        if env_key in env:
            raw = env[env_key]
            overrides[field_name] = _cast(field_name, raw, field_info.annotation)

    return Settings(**overrides)
