"""Tests for src/research_agent/config.py."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from research_agent.config import Settings, load_settings

# ──────────────────────────────────────────────────────────────────────────────
# Default values match SPEC section 3
# ──────────────────────────────────────────────────────────────────────────────


def test_defaults_budgets() -> None:
    s = Settings()
    assert s.MAX_SEARCHES == 8
    assert s.MAX_FETCHES == 12
    assert s.MAX_LLM_CALLS == 30
    assert s.MAX_TOTAL_TOKENS == 200_000
    assert s.MAX_REFINE_LOOPS == 1
    assert s.WALL_TIME_S == 120
    assert s.MAX_WORKERS == 4


def test_defaults_timeouts() -> None:
    s = Settings()
    assert s.SEARCH_TIMEOUT_S == 10
    assert s.FETCH_TIMEOUT_S == 12
    assert s.LLM_TIMEOUT_S == 60
    assert s.MAX_RETRIES == 2
    assert s.BACKOFF_BASE_S == 1.0
    assert s.MAX_CORRECTIONS == 2
    assert s.MAX_REDIRECTS == 3


def test_defaults_fetch() -> None:
    s = Settings()
    assert s.MAX_FETCH_BYTES == 2_000_000
    assert s.MAX_FETCH_TEXT_CHARS == 60_000
    assert s.MIN_PAGE_TEXT_CHARS == 200
    assert s.USER_AGENT == "ResearchAgentBot/0.1 (educational project)"


def test_defaults_planning() -> None:
    s = Settings()
    assert s.MAX_SUBQUESTIONS == 5
    assert s.QUERIES_PER_SUBQUESTION == 2


def test_defaults_search() -> None:
    s = Settings()
    assert s.SEARCH_RESULTS_PER_QUERY == 5
    assert s.SEARCH_PROVIDERS == "tavily,ddgs"


def test_defaults_selection() -> None:
    s = Settings()
    assert s.TOP_K_FETCH_PER_SUBQ == 2
    assert s.SNIPPET_ONLY_TOP_K == 3
    assert s.MIN_SOURCE_SCORE == pytest.approx(0.35)
    assert s.W_RELEVANCE == pytest.approx(0.5)
    assert s.W_DOMAIN == pytest.approx(0.35)
    assert s.W_RECENCY == pytest.approx(0.15)
    assert s.STALE_AFTER_DAYS == 730
    assert s.MIN_SNIPPET_CHARS == 80
    assert s.INJECTION_DROP_THRESHOLD == 3


def test_defaults_extraction() -> None:
    s = Settings()
    assert s.EXTRACT_CHUNK_CHARS == 1200
    assert s.EXTRACT_MAX_CHARS_PER_SOURCE == 6000
    assert s.MAX_EVIDENCE_PER_SOURCE == 3
    assert s.PASSAGE_MIN_WORDS == 5
    assert s.PASSAGE_MAX_WORDS == 40


def test_defaults_synthesis() -> None:
    s = Settings()
    assert s.MAX_EVIDENCE_FOR_SYNTH == 30
    assert s.STRICT_NUMERIC_CHECK is True


def test_defaults_sufficiency() -> None:
    s = Settings()
    assert s.MIN_COVERAGE_TO_ANSWER == pytest.approx(0.5)
    assert s.TARGET_COVERAGE == pytest.approx(0.8)


def test_defaults_confidence() -> None:
    s = Settings()
    assert s.HIGH_MIN_AVG_QUALITY == pytest.approx(0.65)
    assert s.HIGH_MAX_SNIPPET_RATIO == pytest.approx(0.25)
    assert s.HIGH_MIN_DOMAINS == 2
    assert s.MEDIUM_MIN_COVERAGE == pytest.approx(0.7)
    assert s.MEDIUM_MIN_AVG_QUALITY == pytest.approx(0.5)
    assert s.MEDIUM_MAX_CONFLICTS == 1


def test_defaults_secrets_none() -> None:
    s = Settings()
    assert s.TAVILY_API_KEY is None
    assert s.LLM_API_KEY is None
    assert s.LLM_PROVIDER is None
    assert s.LLM_MODEL_CHEAP is None
    assert s.LLM_MODEL_STRONG is None


def test_defaults_paths() -> None:
    s = Settings()
    assert s.LOG_DIR == "logs"
    assert s.RUNS_DIR == "runs"
    assert s.CACHE_DIR == "cache"
    assert s.USE_CACHE is True
    assert s.FAULT_INJECT == ""


# ──────────────────────────────────────────────────────────────────────────────
# Env parsing via load_settings
# ──────────────────────────────────────────────────────────────────────────────


def test_env_int_override() -> None:
    s = load_settings({"MAX_SEARCHES": "15"})
    assert s.MAX_SEARCHES == 15


def test_env_float_override() -> None:
    s = load_settings({"BACKOFF_BASE_S": "2.5"})
    assert s.BACKOFF_BASE_S == pytest.approx(2.5)


def test_env_bool_true_variants() -> None:
    for val in ("true", "True", "TRUE", "1", "yes"):
        s = load_settings({"STRICT_NUMERIC_CHECK": val})
        assert s.STRICT_NUMERIC_CHECK is True, f"failed for {val!r}"


def test_env_bool_false_variants() -> None:
    for val in ("false", "False", "FALSE", "0", "no"):
        s = load_settings({"STRICT_NUMERIC_CHECK": val})
        assert s.STRICT_NUMERIC_CHECK is False, f"failed for {val!r}"


def test_env_secret_parsed() -> None:
    s = load_settings({"TAVILY_API_KEY": "sk-abc123"})
    assert isinstance(s.TAVILY_API_KEY, SecretStr)
    assert s.TAVILY_API_KEY.get_secret_value() == "sk-abc123"


def test_env_str_override() -> None:
    s = load_settings({"LLM_PROVIDER": "openai"})
    assert s.LLM_PROVIDER == "openai"


def test_env_unknown_vars_ignored() -> None:
    """Unknown env vars must not cause an error."""
    s = load_settings({"TOTALLY_UNKNOWN_VAR": "whatever"})
    assert isinstance(s, Settings)


# ──────────────────────────────────────────────────────────────────────────────
# Bad value errors name the variable
# ──────────────────────────────────────────────────────────────────────────────


def test_bad_int_names_variable() -> None:
    with pytest.raises(ValueError, match="MAX_SEARCHES"):
        load_settings({"MAX_SEARCHES": "not-a-number"})


def test_bad_float_names_variable() -> None:
    with pytest.raises(ValueError, match="BACKOFF_BASE_S"):
        load_settings({"BACKOFF_BASE_S": "not-a-float"})


def test_bad_bool_names_variable() -> None:
    with pytest.raises(ValueError, match="STRICT_NUMERIC_CHECK"):
        load_settings({"STRICT_NUMERIC_CHECK": "maybe"})


# ──────────────────────────────────────────────────────────────────────────────
# Secret redaction
# ──────────────────────────────────────────────────────────────────────────────


def test_secret_not_in_repr() -> None:
    s = Settings(TAVILY_API_KEY=SecretStr("super-secret"))  # type: ignore[call-arg]
    r = repr(s)
    assert "super-secret" not in r


def test_secret_not_in_str() -> None:
    s = Settings(TAVILY_API_KEY=SecretStr("super-secret"))  # type: ignore[call-arg]
    assert "super-secret" not in str(s)


def test_secret_not_in_model_dump() -> None:
    s = Settings(TAVILY_API_KEY=SecretStr("super-secret"))  # type: ignore[call-arg]
    d = s.model_dump()
    # model_dump for SecretStr fields returns the SecretStr object itself —
    # the raw value is NOT a plain string.
    val = d["TAVILY_API_KEY"]
    assert isinstance(val, SecretStr)
    assert "super-secret" not in str(val)  # SecretStr.__str__ = '**********'


def test_redacted_dict_masks_set_secret() -> None:
    s = Settings(TAVILY_API_KEY=SecretStr("super-secret"))  # type: ignore[call-arg]
    d = s.redacted_dict()
    assert d["TAVILY_API_KEY"] == "***"


def test_redacted_dict_none_for_unset_secret() -> None:
    s = Settings()
    d = s.redacted_dict()
    assert d["TAVILY_API_KEY"] is None
    assert d["LLM_API_KEY"] is None


def test_redacted_dict_plain_fields_unchanged() -> None:
    s = Settings()
    d = s.redacted_dict()
    assert d["MAX_SEARCHES"] == 8
    assert d["SEARCH_PROVIDERS"] == "tavily,ddgs"


# ──────────────────────────────────────────────────────────────────────────────
# Cross-field validation
# ──────────────────────────────────────────────────────────────────────────────


def test_weight_sum_invalid() -> None:
    with pytest.raises(ValueError, match="W_RELEVANCE"):
        Settings(W_RELEVANCE=0.5, W_DOMAIN=0.5, W_RECENCY=0.5)


def test_weight_sum_valid_with_tolerance() -> None:
    # Floating point representation: 0.5 + 0.35 + 0.15 should pass
    s = Settings()
    assert abs(s.W_RELEVANCE + s.W_DOMAIN + s.W_RECENCY - 1.0) < 1e-6


def test_coverage_ordering_invalid() -> None:
    with pytest.raises(ValueError, match="MIN_COVERAGE_TO_ANSWER"):
        Settings(MIN_COVERAGE_TO_ANSWER=0.9, TARGET_COVERAGE=0.5)


def test_coverage_ordering_equal_valid() -> None:
    s = Settings(MIN_COVERAGE_TO_ANSWER=0.6, TARGET_COVERAGE=0.6)
    assert s.MIN_COVERAGE_TO_ANSWER == pytest.approx(0.6)


def test_budget_field_below_one_invalid() -> None:
    with pytest.raises(ValueError):
        Settings(MAX_SEARCHES=0)


def test_backoff_zero_invalid() -> None:
    with pytest.raises(ValueError):
        Settings(BACKOFF_BASE_S=0.0)
