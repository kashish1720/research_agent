"""JSONL execution tracing and per-node summaries."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SENSITIVE_KEY_RE = re.compile(r"(?:api[_-]?key|access[_-]?token|secret|authorization)", re.I)
_ASSIGNED_SECRET_RE = re.compile(
	r"(?i)(\b(?:api[_-]?key|access[_-]?token|secret|authorization)\s*[:=]\s*)"
	r"(?:Bearer\s+)?([^\s,;\"']+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
_KEY_TOKEN_RE = re.compile(
	r"\b(?:sk-[A-Za-z0-9_-]{8,}|tvly-[A-Za-z0-9_-]{8,}|AIza[A-Za-z0-9_-]{20,})\b"
)


def _sanitize(value: Any, key: str | None = None) -> Any:  # noqa: ANN401
	if key is not None and _SENSITIVE_KEY_RE.search(key):
		return "***"
	if isinstance(value, str):
		value = _ASSIGNED_SECRET_RE.sub(r"\1***", value)
		value = _BEARER_RE.sub("Bearer ***", value)
		value = _KEY_TOKEN_RE.sub("***", value)
		return value[:500]
	if isinstance(value, dict):
		return {str(k): _sanitize(v, str(k)) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_sanitize(item) for item in value]
	if value is None or isinstance(value, (bool, int, float)):
		return value
	return _sanitize(str(value))


class Tracer:
	"""Write sanitized run events to JSONL and retain them in memory."""

	def __init__(self, run_id: str, log_dir: str | Path) -> None:
		self.run_id = run_id
		self.log_dir = Path(log_dir)
		self.log_dir.mkdir(parents=True, exist_ok=True)
		self.path = self.log_dir / f"{run_id}.jsonl"
		self.events: list[dict[str, Any]] = []

	def event(self, kind: str, node: str, **fields: Any) -> dict[str, Any]:  # noqa: ANN401
		"""Record one event using the standard trace fields."""
		details = fields.pop("details", None)
		entry: dict[str, Any] = {
			"ts": datetime.now(UTC).isoformat(),
			"run_id": self.run_id,
			"node": node,
			"event": kind,
			"duration_ms": fields.pop("duration_ms", None),
			"attempt": fields.pop("attempt", None),
			"tokens_in": fields.pop("tokens_in", None),
			"tokens_out": fields.pop("tokens_out", None),
		}
		if fields:
			if details is None:
				details = fields
			elif isinstance(details, dict):
				details = {**details, **fields}
			else:
				details = {"message": details, **fields}
		entry["details"] = _sanitize(details if details is not None else "")
		safe_entry = _sanitize(entry)
		self.events.append(safe_entry)
		with self.path.open("a", encoding="utf-8") as trace_file:
			trace_file.write(json.dumps(safe_entry, ensure_ascii=False) + "\n")
		return safe_entry

	@contextmanager
	def node_span(self, name: str) -> Iterator[None]:
		"""Record a node's start, duration, and any raised error."""
		started_at = time.monotonic()
		self.event("node_start", name)
		try:
			yield
		except Exception as err:
			duration_ms = (time.monotonic() - started_at) * 1000
			self.event(
				"error",
				name,
				duration_ms=duration_ms,
				details={"error_type": type(err).__name__, "message": str(err)},
			)
			self.event("node_end", name, duration_ms=duration_ms)
			raise
		else:
			duration_ms = (time.monotonic() - started_at) * 1000
			self.event("node_end", name, duration_ms=duration_ms)

	def summary(self) -> dict[str, dict[str, int | float]]:
		"""Aggregate elapsed time, tokens, tool calls, retries, and failures per node."""
		totals: dict[str, dict[str, int | float]] = {}
		for event in self.events:
			node = str(event["node"])
			stats = totals.setdefault(
				node,
				{
					"duration_ms": 0.0,
					"tokens_in": 0,
					"tokens_out": 0,
					"tool_calls": 0,
					"retries": 0,
					"failures": 0,
				},
			)
			if event["event"] == "node_end":
				stats["duration_ms"] += event["duration_ms"] or 0.0
			stats["tokens_in"] += event["tokens_in"] or 0
			stats["tokens_out"] += event["tokens_out"] or 0
			if event["event"] == "tool_call":
				stats["tool_calls"] += 1
			elif event["event"] == "retry":
				stats["retries"] += 1
			elif event["event"] == "error":
				stats["failures"] += 1
		return totals

	def render_summary(self) -> str:
		"""Render the summary as readable plain text."""
		header = "Node | time ms | tokens in/out | tool calls | retries | failures"
		rows = [header]
		for node, stats in sorted(self.summary().items()):
			rows.append(
				f"{node} | {stats['duration_ms']:.1f} | "
				f"{stats['tokens_in']}/{stats['tokens_out']} | "
				f"{stats['tool_calls']} | {stats['retries']} | {stats['failures']}"
			)
		return "\n".join(rows)

