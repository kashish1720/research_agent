"""Pydantic schemas and data models (SPEC section 4)."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────


class Strategy(StrEnum):
    """Search strategy for a sub-question."""

    search_only = "search_only"
    search_and_fetch = "search_and_fetch"


class ErrorClass(StrEnum):
    """Classification of a tool error."""

    TRANSIENT = "TRANSIENT"
    CLIENT = "CLIENT"
    CONTENT = "CONTENT"
    SCHEMA = "SCHEMA"
    SECURITY = "SECURITY"
    BUDGET = "BUDGET"
    UNKNOWN = "UNKNOWN"


class FetchStatus(StrEnum):
    """Terminal outcome of a fetch attempt."""

    OK = "OK"
    EMPTY = "EMPTY"
    BLOCKED = "BLOCKED"
    NOT_FOUND = "NOT_FOUND"
    HTTP_ERROR = "HTTP_ERROR"
    NON_HTML = "NON_HTML"
    TOO_LARGE = "TOO_LARGE"
    SECURITY_BLOCKED = "SECURITY_BLOCKED"


class Confidence(StrEnum):
    """Confidence level of a final answer."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RunStatus(StrEnum):
    """Overall run outcome."""

    DONE = "DONE"
    DEGRADED = "DEGRADED"
    FAILED_GRACEFULLY = "FAILED_GRACEFULLY"


class Reason(StrEnum):
    """Reason codes attached to a run."""

    UNCOVERED_SUBQUESTIONS = "UNCOVERED_SUBQUESTIONS"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    SNIPPET_ONLY_EVIDENCE = "SNIPPET_ONLY_EVIDENCE"
    FINDINGS_DROPPED = "FINDINGS_DROPPED"
    PROVIDER_FALLBACK = "PROVIDER_FALLBACK"
    VERIFICATION_UNAVAILABLE = "VERIFICATION_UNAVAILABLE"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    NODE_ERROR = "NODE_ERROR"
    PLAN_FALLBACK = "PLAN_FALLBACK"
    NOT_ANSWERABLE = "NOT_ANSWERABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    ALL_FINDINGS_UNSUPPORTED = "ALL_FINDINGS_UNSUPPORTED"


# ──────────────────────────────────────────────────────────────────────────────
# Helper regex
# ──────────────────────────────────────────────────────────────────────────────

_SUB_Q_ID_RE = re.compile(r"^q[0-9]+$")
_URL_PREFIX_RE = re.compile(r"^https?://", re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────────────
# Shared validator
# ──────────────────────────────────────────────────────────────────────────────


def _validate_query_text(v: str) -> str:
    """Strip, enforce 3..200 chars, reject URL-like strings.

    Applies to SearchArgs.query, SubQuestion.queries items, RefinedQuery.queries items.
    """
    if not isinstance(v, str):
        raise ValueError("query must be a string")
    v = v.strip()
    if len(v) < 3 or len(v) > 200:
        raise ValueError("query must be between 3 and 200 characters")
    if _URL_PREFIX_RE.match(v):
        raise ValueError("query must not be a URL (starts with http:// or https://)")
    return v


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────


class SearchArgs(BaseModel):
    """Arguments for a single search call."""

    model_config = ConfigDict(extra="forbid")

    query: str
    max_results: int = Field(default=5, ge=1, le=10)
    depth: Literal["basic", "advanced"] = "basic"

    @field_validator("query", mode="before")
    @classmethod
    def _validate_query(cls, v: object) -> str:
        if not isinstance(v, str):
            raise ValueError("query must be a string")
        return _validate_query_text(v)


class SubQuestion(BaseModel):
    """A single sub-question in a research plan."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    queries: list[str] = Field(..., min_length=1, max_length=2)
    strategy: Strategy
    time_sensitive: bool
    rationale: str

    @field_validator("id", mode="after")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not _SUB_Q_ID_RE.match(v):
            raise ValueError(f"sub-question id must match ^q[0-9]+$, got {v!r}")
        return v

    @field_validator("text", "rationale", mode="after")
    @classmethod
    def _non_blank_text_rationale(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field must not be blank")
        return v

    @field_validator("queries", mode="before")
    @classmethod
    def _validate_queries(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            raise ValueError("queries must be a list")
        return [_validate_query_text(item) for item in v]


class Plan(BaseModel):
    """LLM-facing research plan."""

    model_config = ConfigDict(extra="forbid")

    restated_question: str
    is_answerable: bool
    clarification_note: str | None = None
    assumptions: list[str]
    sub_questions: list[SubQuestion]

    @model_validator(mode="after")
    def _validate_plan_consistency(self) -> Plan:
        if self.is_answerable:
            n = len(self.sub_questions)
            if n < 1:
                raise ValueError(
                    "An answerable plan must have at least 1 sub-question"
                )
            # NOTE: The upper limit MAX_SUBQUESTIONS is enforced by the plan node
            # via TRUNCATION, NOT by the schema (SPEC v3.1 section 4).
            # Unique ids
            ids = [sq.id for sq in self.sub_questions]
            if len(ids) != len(set(ids)):
                raise ValueError("sub-question ids must be unique within a plan")
        else:
            note = self.clarification_note
            if not note or not note.strip():
                raise ValueError(
                    "A non-answerable plan must include a non-blank clarification_note"
                )
        return self


class SearchResult(BaseModel):
    """A single result returned by a search provider.

    Stays permissive (no extra='forbid') because it is built from provider data.
    """

    url: str
    title: str
    snippet: str
    score: float | None = None
    published_date: str | None = None
    provider: str
    sub_question_id: str | None = None
    query: str | None = None


class ToolError(BaseModel):
    """Structured description of a tool failure."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    error_class: ErrorClass
    message: str
    http_status: int | None = None
    attempt: int = 1

    @property
    def retriable(self) -> bool:
        """True when the error class is TRANSIENT."""
        return self.error_class is ErrorClass.TRANSIENT


class FetchResult(BaseModel):
    """Outcome of fetching a single URL.

    Stays permissive (no extra='forbid') because it is built from provider data.
    """

    url: str
    final_url: str | None = None
    status: FetchStatus
    text: str = ""
    title: str | None = None
    published_date: str | None = None
    http_status: int | None = None
    truncated: bool = False
    error: ToolError | None = None


class Source(BaseModel):
    """A processed source for a research run."""

    model_config = ConfigDict(extra="forbid")

    id: str
    url: str
    domain: str
    title: str
    published_date: str | None = None
    quality_score: float = Field(..., ge=0.0, le=1.0)
    fetched: bool
    snippet_only: bool
    flags: list[str] = Field(default_factory=list)
    sub_question_ids: list[str] = Field(default_factory=list)
    # Excluded from serialization and repr per SPEC
    text: str | None = Field(default=None, exclude=True, repr=False)


class EvidenceItem(BaseModel):
    """A single piece of evidence extracted from a source."""

    model_config = ConfigDict(extra="forbid")

    id: str
    sub_question_id: str
    source_id: str
    domain: str
    published_date: str | None = None
    passage: str
    claim: str
    snippet_only: bool

    @field_validator("passage", mode="after")
    @classmethod
    def _passage_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("passage must not be empty")
        return v


class ExtractionItem(BaseModel):
    """LLM-facing: a single item in an extraction response."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    passage: str
    claim: str


class ExtractionOutput(BaseModel):
    """LLM-facing: full extraction response."""

    model_config = ConfigDict(extra="forbid")

    items: list[ExtractionItem]


class Conflict(BaseModel):
    """A detected conflict between evidence items."""

    model_config = ConfigDict(extra="forbid")

    topic: str
    description: str
    evidence_ids: list[str]

    @field_validator("evidence_ids", mode="after")
    @classmethod
    def _at_least_two_unique(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("evidence_ids must contain at least 2 entries")
        if len(v) != len(set(v)):
            raise ValueError("evidence_ids must be unique")
        return v


class FindingDraft(BaseModel):
    """LLM-facing: a single drafted finding."""

    model_config = ConfigDict(extra="forbid")

    statement: str
    evidence_ids: list[str]

    @field_validator("evidence_ids", mode="after")
    @classmethod
    def _non_empty_unique(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("evidence_ids must not be empty")
        if len(v) != len(set(v)):
            raise ValueError("evidence_ids must be unique")
        return v


class SynthesisDraft(BaseModel):
    """LLM-facing: full synthesis response."""

    model_config = ConfigDict(extra="forbid")

    short_answer: str
    findings: list[FindingDraft]
    conflicts: list[Conflict]
    limitations: list[str]
    next_steps: list[str]


class Finding(BaseModel):
    """A verified finding."""

    model_config = ConfigDict(extra="forbid")

    statement: str
    evidence_ids: list[str]
    support: Literal["supported", "partial", "unchecked"] = "unchecked"

    @field_validator("evidence_ids", mode="after")
    @classmethod
    def _non_empty_unique(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("evidence_ids must not be empty")
        if len(v) != len(set(v)):
            raise ValueError("evidence_ids must be unique")
        return v


class EntailmentItem(BaseModel):
    """LLM-facing: entailment verdict for a single finding."""

    model_config = ConfigDict(extra="forbid")

    index: int
    verdict: Literal["supported", "partial", "unsupported"]
    reason: str


class EntailmentOutput(BaseModel):
    """LLM-facing: full entailment response."""

    model_config = ConfigDict(extra="forbid")

    items: list[EntailmentItem]


class RefinedQuery(BaseModel):
    """LLM-facing: refined queries for a single sub-question."""

    model_config = ConfigDict(extra="forbid")

    sub_question_id: str
    queries: list[str] = Field(..., min_length=1, max_length=2)

    @field_validator("queries", mode="before")
    @classmethod
    def _validate_queries(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            raise ValueError("queries must be a list")
        return [_validate_query_text(item) for item in v]


class RefinedQueries(BaseModel):
    """LLM-facing: full refine response."""

    model_config = ConfigDict(extra="forbid")

    items: list[RefinedQuery]


class SubQuestionCoverage(BaseModel):
    """Coverage details for a single sub-question."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    evidence_count: int
    independent_domains: int


class Coverage(BaseModel):
    """Overall coverage summary."""

    model_config = ConfigDict(extra="forbid")

    total: int
    covered: int
    per_subquestion: list[SubQuestionCoverage]


class ConfidenceReport(BaseModel):
    """Rule-based confidence assessment."""

    model_config = ConfigDict(extra="forbid")

    level: Confidence
    coverage_ratio: float = Field(..., ge=0.0, le=1.0)
    avg_source_quality: float = Field(..., ge=0.0, le=1.0)
    min_independent_domains: int = Field(..., ge=0)
    unresolved_conflicts: int = Field(..., ge=0)
    snippet_only_ratio: float = Field(..., ge=0.0, le=1.0)
    verification_complete: bool
    reasons: list[str]


class UsageSummary(BaseModel):
    """Resource usage for a run."""

    model_config = ConfigDict(extra="forbid")

    llm_calls: int
    input_tokens: int
    output_tokens: int
    searches: int
    fetches: int
    tool_retries: int
    latency_s: float
    est_cost_usd: float | None = None


class FinalAnswer(BaseModel):
    """Complete structured answer from the research agent."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    user_question: str
    short_answer: str
    key_findings: list[Finding]
    sources_used: list[Source]
    confidence: ConfidenceReport
    limitations_and_assumptions: list[str]
    suggested_next_steps: list[str]
    conflicts: list[Conflict]
    coverage: Coverage
    status: RunStatus
    reasons: list[Reason]
    usage: UsageSummary


class AgentState(BaseModel):
    """Mutable state threaded through the LangGraph graph."""

    model_config = ConfigDict(extra="forbid")

    question: str
    today: str
    plan: Plan | None = None
    search_results: dict[str, list[SearchResult]] = Field(default_factory=dict)
    sources: dict[str, Source] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    pending_sq_ids: list[str] = Field(default_factory=list)
    refine_loops: int = 0
    draft: SynthesisDraft | None = None
    findings: list[Finding] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    short_answer: str | None = None
    final: FinalAnswer | None = None
    reasons: list[Reason] = Field(default_factory=list)
    abort: bool = False
    errors: list[str] = Field(default_factory=list)
    stats: dict[str, int] = Field(default_factory=dict)
    route: Literal["refine", "synthesize", "insufficient"] | None = None
