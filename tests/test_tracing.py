"""Tests for JSONL tracing and readable summaries."""

from __future__ import annotations

import json

import pytest

from research_agent.tracing import Tracer


def test_events_are_valid_jsonl_with_standard_fields(tmp_path) -> None:  # type: ignore[no-untyped-def]
    tracer = Tracer("run-1", tmp_path / "nested" / "logs")
    tracer.event("tool_call", "search", attempt=1, tokens_in=4, tokens_out=2, details="ok")

    lines = tracer.path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[0])
    assert event["run_id"] == "run-1"
    assert event["node"] == "search"
    assert event["event"] == "tool_call"
    assert event["attempt"] == 1
    assert event["tokens_in"] == 4
    assert event["tokens_out"] == 2
    assert event["details"] == "ok"
    assert tracer.events == [event]


def test_strings_are_truncated_and_secrets_redacted(tmp_path) -> None:  # type: ignore[no-untyped-def]
    tracer = Tracer("run-2", tmp_path)
    tracer.event(
        "note",
        "plan",
        details={
            "long": "x" * 700,
            "api_key": "plain-secret",
            "authorization": "Bearer token-value",
            "message": "using Bearer abc.def-ghi and sk-1234567890abcdef",
        },
    )

    event = json.loads(tracer.path.read_text(encoding="utf-8"))
    details = event["details"]
    assert len(details["long"]) == 500
    assert details["api_key"] == "***"
    assert details["authorization"] == "***"
    assert "token-value" not in json.dumps(event)
    assert "abc.def-ghi" not in json.dumps(event)
    assert "sk-1234567890abcdef" not in json.dumps(event)


def test_node_span_records_duration_and_summary(tmp_path) -> None:  # type: ignore[no-untyped-def]
    tracer = Tracer("run-3", tmp_path)
    with tracer.node_span("extract"):
        tracer.event("llm_call", "extract", tokens_in=12, tokens_out=5)
        tracer.event("tool_call", "extract")
        tracer.event("retry", "extract")

    summary = tracer.summary()["extract"]
    assert summary["duration_ms"] >= 0
    assert summary["tokens_in"] == 12
    assert summary["tokens_out"] == 5
    assert summary["tool_calls"] == 1
    assert summary["retries"] == 1
    assert "extract |" in tracer.render_summary()


def test_node_span_logs_error_then_reraises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    tracer = Tracer("run-4", tmp_path)

    with pytest.raises(RuntimeError, match="node failed"):
        with tracer.node_span("verify"):
            raise RuntimeError("node failed")

    event_names = [event["event"] for event in tracer.events]
    assert event_names == ["node_start", "error", "node_end"]
    assert tracer.summary()["verify"]["failures"] == 1