"""Typed structured LLM clients for Groq and deterministic tests."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Generic, Literal, Protocol, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from research_agent.budget import Budget
from research_agent.cache import DiskCache
from research_agent.config import Settings
from research_agent.errors import BudgetExceeded, ToolFailure
from research_agent.schemas import ErrorClass, ToolError
from research_agent.tracing import Tracer

T = TypeVar("T", bound=BaseModel)
Role = Literal["cheap", "strong"]
_DEFAULT_MODELS = {
	"cheap": "openai/gpt-oss-20b",
	"strong": "openai/gpt-oss-120b",
}
_MAX_COMPLETION_TOKENS = 8192
_UNSUPPORTED_SCHEMA_TERMS = ("json_schema", "json schema", "response_format", "strict")
_DEBUG_BEARER_RE = re.compile(r"(?i)\bBearer\s+\S+")
_DEBUG_SECRET_RE = re.compile(
	r"(?i)(?:\b(?:sk-[A-Za-z0-9_-]{8,}|gsk_[A-Za-z0-9_-]{8,})|"
	r"\b(?:api[_-]?key|token)[\"']?\s*[:=]\s*[\"']?[^\"'\s,}]+)"
)
_LOGGER = logging.getLogger(__name__)
_SIMPLIFIED_SCHEMA_OMIT = {
	"$schema",
	"default",
	"description",
	"examples",
	"exclusiveMaximum",
	"exclusiveMinimum",
	"format",
	"maximum",
	"maxItems",
	"maxLength",
	"minimum",
	"minItems",
	"minLength",
	"pattern",
	"title",
}


@dataclass(frozen=True)
class LLMResult(Generic[T]):
	"""Parsed structured output and response usage metadata."""

	parsed: T
	input_tokens: int
	output_tokens: int
	model: str
	latency_ms: float


class GenerationRetryRequired(Exception):
	"""Signals a schema-generation 400 that should be retried in JSON mode."""


class LLMOutputError(ValueError):
	"""A content/schema failure retaining token usage from the response."""

	def __init__(self, message: str, input_tokens: int, output_tokens: int) -> None:
		self.input_tokens = input_tokens
		self.output_tokens = output_tokens
		super().__init__(message)


class LLMClient(Protocol):
	"""A tool-less structured-output LLM interface."""

	def structured(
		self,
		*,
		role: Role,
		system: str,
		user: str,
		schema: type[T],
		validate: Callable[[T], None] | None = None,
	) -> LLMResult[T]:
		"""Return a validated structured completion."""


def _simplified_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
	"""Inline Pydantic references and omit constraints validated locally."""
	raw_schema = schema.model_json_schema(mode="validation")
	definitions = raw_schema.pop("$defs", {})

	def simplify(value: Any, parents: frozenset[str] = frozenset()) -> Any:  # noqa: ANN401
		if isinstance(value, list):
			return [simplify(item, parents) for item in value]
		if not isinstance(value, dict):
			return value
		reference = value.get("$ref")
		if isinstance(reference, str) and reference.startswith("#/$defs/"):
			definition_name = reference.rsplit("/", 1)[-1]
			if definition_name in parents:
				return {"type": "object"}
			definition = definitions.get(definition_name)
			if isinstance(definition, dict):
				merged = {key: item for key, item in value.items() if key != "$ref"}
				merged.update(definition)
				merged.update({key: item for key, item in value.items() if key != "$ref"})
				return simplify(merged, parents | {definition_name})
		result = {
			key: simplify(item, parents)
			for key, item in value.items()
			if key not in _SIMPLIFIED_SCHEMA_OMIT and key != "$ref"
		}
		if result.get("type") == "object" and isinstance(result.get("properties"), dict):
			result["required"] = list(result["properties"])
			result["additionalProperties"] = False
		return result

	simplified = simplify(raw_schema)
	if not isinstance(simplified, dict):
		raise TypeError("Pydantic returned a non-object JSON schema")
	return simplified


def _schema_response_format(schema: type[BaseModel]) -> dict[str, Any]:
	name = re.sub(r"[^a-zA-Z0-9_-]", "_", schema.__name__).lower()
	return {
		"type": "json_schema",
		"json_schema": {
			"name": name,
			"strict": True,
			"schema": _simplified_json_schema(schema),
		},
	}


def _retry_after_seconds(error: Exception) -> float | None:
	response = getattr(error, "response", None)
	headers = getattr(response, "headers", None)
	if headers is None:
		return None
	value = headers.get("retry-after")
	if value is None:
		return None
	try:
		return max(0.0, float(value))
	except (TypeError, ValueError):
		try:
			retry_at = parsedate_to_datetime(str(value))
			if retry_at.tzinfo is None:
				retry_at = retry_at.replace(tzinfo=UTC)
			return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
		except (TypeError, ValueError, OverflowError):
			return None


def _map_sdk_error(error: Exception) -> ToolFailure:
	response = getattr(error, "response", None)
	status = getattr(error, "status_code", None)
	if status is None:
		status = getattr(response, "status_code", None)
	if isinstance(error, openai.RateLimitError) or status == 429:
		failure = ToolFailure(
			ToolError(
				tool="llm",
				error_class=ErrorClass.TRANSIENT,
				message="Groq rate limit exceeded",
				http_status=429,
			)
		)
		retry_after = _retry_after_seconds(error)
		if retry_after is not None:
			failure.retry_after_s = retry_after
		return failure
	if isinstance(error, (openai.APITimeoutError, openai.APIConnectionError)):
		return ToolFailure(
			ToolError(
				tool="llm",
				error_class=ErrorClass.TRANSIENT,
				message=f"Groq transport failed ({type(error).__name__})",
			)
		)
	if status is not None and status >= 500 or isinstance(error, openai.InternalServerError):
		return ToolFailure(
			ToolError(
				tool="llm",
				error_class=ErrorClass.TRANSIENT,
				message="Groq returned a server error",
				http_status=status if isinstance(status, int) else None,
			)
		)
	if isinstance(error, (openai.AuthenticationError, openai.BadRequestError)) or (
		isinstance(status, int) and 400 <= status < 500
	):
		return ToolFailure(
			ToolError(
				tool="llm",
				error_class=ErrorClass.CLIENT,
				message=f"Groq rejected the request ({type(error).__name__})",
				http_status=status if isinstance(status, int) else None,
			)
		)
	return ToolFailure(
		ToolError(
			tool="llm",
			error_class=ErrorClass.UNKNOWN,
			message=f"Groq SDK failed ({type(error).__name__})",
		)
	)


def _is_schema_format_rejection(error: openai.BadRequestError) -> bool:
	message = str(error).lower()
	return any(term in message for term in _UNSUPPORTED_SCHEMA_TERMS)


def _is_generation_failure(error: openai.BadRequestError) -> bool:
	body = getattr(error, "body", None)
	if _debug_field(body, "failed_generation") is not None:
		return True
	message = f"{error} {body}".lower()
	return (
		"failed_generation" in message
		or "does not match the expected schema" in message
		or _is_schema_format_rejection(error)
	)


def _salvage_failed_generation(
	error: openai.BadRequestError,
	schema: type[T],
	model: str,
	started: float,
) -> LLMResult[T] | None:
	failed_generation = _debug_field(getattr(error, "body", None), "failed_generation")
	if not isinstance(failed_generation, str):
		return None
	try:
		parsed = schema.model_validate_json(failed_generation)
	except (ValidationError, ValueError, TypeError):
		return None
	usage = _debug_field(getattr(error, "body", None), "usage")
	input_tokens = _debug_field(usage, "prompt_tokens")
	output_tokens = _debug_field(usage, "completion_tokens")
	return LLMResult(
		parsed=parsed,
		input_tokens=input_tokens if isinstance(input_tokens, int) else 0,
		output_tokens=output_tokens if isinstance(output_tokens, int) else 0,
		model=model,
		latency_ms=(time.perf_counter() - started) * 1000,
	)


def _debug_field(value: object, key: str) -> object | None:
	if isinstance(value, Mapping):
		if key in value:
			return value[key]
		for nested in value.values():
			found = _debug_field(nested, key)
			if found is not None:
				return found
	elif isinstance(value, (list, tuple)):
		for nested in value:
			found = _debug_field(nested, key)
			if found is not None:
				return found
	return None


def _debug_usage(usage: object | None) -> dict[str, int | None] | None:
	if usage is None:
		return None
	completion_details = _debug_field(usage, "completion_tokens_details")
	values: dict[str, int | None] = {}
	for key in ("prompt_tokens", "completion_tokens"):
		value = _debug_field(usage, key)
		values[key] = value if isinstance(value, int) else None
	reasoning_tokens = _debug_field(completion_details, "reasoning_tokens")
	values["reasoning_tokens"] = (
		reasoning_tokens if isinstance(reasoning_tokens, int) else None
	)
	return values


def _debug_safe_text(value: object) -> str:
	text = str(value).replace('\\"', '"')
	text = _DEBUG_BEARER_RE.sub("Bearer ***", text)
	return _DEBUG_SECRET_RE.sub("***", text)[:300]


def _validation_error_text(error: Exception) -> str:
	if isinstance(error, ValidationError):
		messages = [
			f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
			for item in error.errors(include_input=False)
		]
		return _debug_safe_text("; ".join(messages))
	return _debug_safe_text(error)


def _debug_log(
	*,
	finish_reason: object | None,
	usage: object | None,
	max_completion_tokens: int,
	failed_generation: object | None,
) -> None:
	if os.environ.get("LLM_DEBUG") != "1":
		return
	failed_excerpt = (
		_debug_safe_text(failed_generation) if failed_generation is not None else None
	)
	_LOGGER.warning(
		"LLM_DEBUG finish_reason=%s usage=%s max_completion_tokens=%s failed_generation=%s",
		finish_reason,
		_debug_usage(usage),
		max_completion_tokens,
		failed_excerpt,
	)


def _debug_bad_request(error: openai.BadRequestError, max_completion_tokens: int) -> None:
	body = getattr(error, "body", None)
	if isinstance(body, str):
		try:
			body = json.loads(body)
		except json.JSONDecodeError:
			body = {"message": body}
	_debug_log(
		finish_reason=_debug_field(body, "finish_reason"),
		usage=_debug_field(body, "usage"),
		max_completion_tokens=max_completion_tokens,
		failed_generation=_debug_field(body, "failed_generation"),
	)


def _debug_completion(response: object, max_completion_tokens: int) -> None:
	choices = getattr(response, "choices", None)
	choice = choices[0] if isinstance(choices, (list, tuple)) and choices else None
	_debug_log(
		finish_reason=getattr(choice, "finish_reason", None),
		usage=getattr(response, "usage", None),
		max_completion_tokens=max_completion_tokens,
		failed_generation=_debug_field(response, "failed_generation"),
	)


class GroqClient:
	"""Use Groq's OpenAI-compatible Chat Completions API."""

	def __init__(self, settings: Settings, client: Any | None = None) -> None:  # noqa: ANN401
		self.settings = settings
		if client is not None:
			self.client = client
			return
		if settings.LLM_API_KEY is None:
			raise ValueError("LLM_API_KEY is required to create the Groq client")
		self.client = OpenAI(
			api_key=settings.LLM_API_KEY.get_secret_value(),
			base_url="https://api.groq.com/openai/v1",
			timeout=settings.LLM_TIMEOUT_S,
			max_retries=0,
		)

	def model_for(self, role: str) -> str:
		if role not in _DEFAULT_MODELS:
			raise ValueError(f"unknown LLM role: {role}")
		configured = (
			self.settings.LLM_MODEL_CHEAP if role == "cheap" else self.settings.LLM_MODEL_STRONG
		)
		return configured or _DEFAULT_MODELS[role]

	def structured(
		self,
		*,
		role: Role,
		system: str,
		user: str,
		schema: type[T],
		validate: Callable[[T], None] | None = None,
	) -> LLMResult[T]:
		del validate
		return self._structured(
			role=role,
			system=system,
			user=user,
			schema=schema,
			json_mode=False,
		)

	def structured_json_mode(
		self,
		*,
		role: Role,
		system: str,
		user: str,
		schema: type[T],
	) -> LLMResult[T]:
		"""Issue one JSON-mode request with the schema included in the prompt."""
		return self._structured(
			role=role,
			system=system,
			user=user,
			schema=schema,
			json_mode=True,
		)

	def _structured(
		self,
		*,
		role: Role,
		system: str,
		user: str,
		schema: type[T],
		json_mode: bool,
	) -> LLMResult[T]:
		model = self.model_for(role)
		started = time.perf_counter()
		system = (
			f"{system.rstrip()}\n\nRespond directly with exactly one JSON object matching the "
			"requested schema. Do not include reasoning, analysis, markdown, or surrounding text."
		)
		if json_mode:
			system = (
				f"{system}\n\nAnswer directly with exactly one JSON object and no reasoning, "
				f"markdown, or surrounding text. The object must match this JSON Schema:\n"
				f"{json.dumps(_simplified_json_schema(schema), ensure_ascii=True)}"
			)
		messages = [
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		]
		call_options: dict[str, Any] = {
			"model": model,
			"messages": messages,
			"temperature": 0,
			"reasoning_effort": "low",
			"max_completion_tokens": _MAX_COMPLETION_TOKENS,
			"timeout": self.settings.LLM_TIMEOUT_S,
		}
		try:
			response = self.client.chat.completions.create(
				**call_options,
				response_format=(
					{"type": "json_object"} if json_mode else _schema_response_format(schema)
				),
			)
		except openai.BadRequestError as error:
			_debug_bad_request(error, call_options["max_completion_tokens"])
			if _is_generation_failure(error):
				salvaged = _salvage_failed_generation(error, schema, model, started)
				if salvaged is not None:
					return salvaged
				if json_mode:
					raise ValueError("JSON-mode generation failed schema validation") from error
				raise GenerationRetryRequired("retry schema generation in JSON mode") from error
			if _is_schema_format_rejection(error) and not json_mode:
				raise GenerationRetryRequired(
					"retry unsupported JSON schema in JSON mode"
				) from error
			raise _map_sdk_error(error) from error
		except ToolFailure:
			raise
		except Exception as error:
			raise _map_sdk_error(error) from error
		_debug_completion(response, call_options["max_completion_tokens"])

		choice = response.choices[0] if response.choices else None
		message = choice.message if choice is not None else None
		usage = getattr(response, "usage", None)
		input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
		output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
		refusal = getattr(message, "refusal", None)
		if refusal:
			raise LLMOutputError(
				"Groq refused the structured request", input_tokens, output_tokens
			)
		content = getattr(message, "content", None)
		if not isinstance(content, str) or not content.strip():
			raise LLMOutputError(
				"Groq returned empty structured output", input_tokens, output_tokens
			)
		try:
			parsed = schema.model_validate_json(content)
		except (ValidationError, ValueError, TypeError) as error:
			raise LLMOutputError(str(error), input_tokens, output_tokens) from error
		latency_ms = (time.perf_counter() - started) * 1000
		return LLMResult(
			parsed=parsed,
			input_tokens=input_tokens,
			output_tokens=output_tokens,
			model=str(getattr(response, "model", None) or model),
			latency_ms=latency_ms,
		)


class ResilientLLM:
	"""Retry transient failures and correct schema-invalid completions."""

	def __init__(
		self,
		inner: LLMClient,
		settings: Settings,
		budget: Budget,
		tracer: Tracer,
		sleep_fn: Callable[[float], None] = time.sleep,
	) -> None:
		self.inner = inner
		self.settings = settings
		self.budget = budget
		self.tracer = tracer
		self.sleep_fn = sleep_fn
		self.cache = DiskCache(
			settings.CACHE_DIR,
			enabled=settings.USE_CACHE and not settings.FAULT_INJECT.strip(),
		)

	def _model_for(self, role: str) -> str:
		model_for = getattr(self.inner, "model_for", None)
		if callable(model_for):
			return str(model_for(role))
		return "fake" if isinstance(self.inner, FakeLLMClient) else role

	def _cost(self, role: str, input_tokens: int, output_tokens: int) -> float | None:
		suffix = "CHEAP" if role == "cheap" else "STRONG"
		input_rate = getattr(self.settings, f"PRICE_IN_PER_MTOK_{suffix}")
		output_rate = getattr(self.settings, f"PRICE_OUT_PER_MTOK_{suffix}")
		if input_rate is None and output_rate is None:
			return None
		return (
			(input_rate or 0.0) * input_tokens + (output_rate or 0.0) * output_tokens
		) / 1_000_000

	def structured(
		self,
		*,
		role: Role,
		system: str,
		user: str,
		schema: type[T],
		validate: Callable[[T], None] | None = None,
	) -> LLMResult[T]:
		model = self._model_for(role)
		cache_inputs = {
			"role": role,
			"model": model,
			"system": system,
			"user": user,
			"schema": schema.__name__,
		}
		found, cached_value = self.cache.load("llm", cache_inputs)
		if found:
			try:
				cached_parsed = schema.model_validate(cached_value)
				if validate is not None:
					validate(cached_parsed)
			except (ValidationError, ValueError):
				pass
			else:
				self.tracer.event(
					"note",
					"llm",
					details={"cache_hit": True, "role": role, "model": model},
				)
				return LLMResult(
					parsed=cached_parsed,
					input_tokens=0,
					output_tokens=0,
					model=model,
					latency_ms=0.0,
				)

		retries = 0
		corrections = 0
		attempt = 0
		current_user = user
		json_mode_next = False
		json_fallback_used = False
		last_validation_error = ""
		while True:
			if not self.budget.can_llm():
				resource = (
					"llm_calls"
					if self.budget.llm_calls >= self.settings.MAX_LLM_CALLS
					else "tokens"
				)
				raise BudgetExceeded(resource)  # type: ignore[arg-type]
			self.budget.consume_llm(0, 0)
			attempt += 1
			started = time.perf_counter()
			use_json_mode = json_mode_next
			json_mode_next = False
			result: LLMResult[T] | None = None
			error: Exception | None = None
			input_tokens = 0
			output_tokens = 0
			tokens_accounted = False
			try:
				json_mode_method = getattr(self.inner, "structured_json_mode", None)
				if use_json_mode and callable(json_mode_method):
					result = json_mode_method(
						role=role,
						system=system,
						user=current_user,
						schema=schema,
					)
				else:
					request_system = system
					if use_json_mode:
						request_system += (
							"\n\nAnswer directly with exactly one JSON object and no reasoning, "
							"markdown, or surrounding text. The object must match this "
							"JSON Schema:\n"
							f"{json.dumps(_simplified_json_schema(schema), ensure_ascii=True)}"
						)
					result = self.inner.structured(
						role=role,
						system=request_system,
						user=current_user,
						schema=schema,
					)
				input_tokens = result.input_tokens
				output_tokens = result.output_tokens
				self.budget.input_tokens += input_tokens
				self.budget.output_tokens += output_tokens
				tokens_accounted = True
				parsed_value = (
					result.parsed
					if isinstance(result.parsed, schema)
					else schema.model_validate(result.parsed)
				)
				if validate is not None:
					validate(parsed_value)
				parsed_result = LLMResult(
					parsed=parsed_value,
					input_tokens=input_tokens,
					output_tokens=output_tokens,
					model=result.model,
					latency_ms=result.latency_ms,
				)
			except Exception as caught_error:
				error = caught_error
				input_tokens = int(getattr(caught_error, "input_tokens", 0) or 0)
				output_tokens = int(getattr(caught_error, "output_tokens", 0) or 0)
				if input_tokens or output_tokens:
					self.budget.input_tokens += input_tokens
					self.budget.output_tokens += output_tokens
					tokens_accounted = True
				parsed_result = None
			finally:
				elapsed_ms = (time.perf_counter() - started) * 1000
				trace_fields: dict[str, Any] = {
					"role": role,
					"model": model,
					"attempt": attempt,
					"tokens_in": input_tokens,
					"tokens_out": output_tokens,
					"duration_ms": elapsed_ms,
					"latency_ms": elapsed_ms,
				}
				cost = self._cost(role, input_tokens, output_tokens)
				if cost is not None:
					trace_fields["cost_usd"] = cost
				if error is not None:
					trace_fields["error_type"] = type(error).__name__
					if isinstance(error, (ValidationError, ValueError, TypeError)):
						last_validation_error = _validation_error_text(error)
						trace_fields["validation_error"] = last_validation_error
				self.tracer.event("llm_call", "llm", **trace_fields)

			if error is None:
				assert parsed_result is not None
				if (
					self.budget.input_tokens + self.budget.output_tokens
					> self.settings.MAX_TOTAL_TOKENS
				):
					raise BudgetExceeded("tokens")
				self.cache.store(
					"llm",
					cache_inputs,
					parsed_result.parsed.model_dump(mode="json"),
				)
				return parsed_result

			if tokens_accounted and (
				self.budget.input_tokens + self.budget.output_tokens
				> self.settings.MAX_TOTAL_TOKENS
			):
				raise BudgetExceeded("tokens") from error

			if isinstance(error, ToolFailure):
				if (
					error.error.error_class is ErrorClass.TRANSIENT
					and retries < self.settings.MAX_RETRIES
				):
					retries += 1
					self.budget.tool_retries += 1
					exponential = self.settings.BACKOFF_BASE_S * (2 ** (retries - 1))
					delay = exponential + random.uniform(0.0, exponential * 0.1)
					retry_after = getattr(error, "retry_after_s", None)
					if isinstance(retry_after, (int, float)):
						delay = max(delay, float(retry_after))
					self.tracer.event(
						"retry",
						"llm",
						attempt=attempt,
						details={"delay_s": delay, "role": role},
					)
					self.sleep_fn(delay)
					continue
				raise error

			if isinstance(error, GenerationRetryRequired):
				if not json_fallback_used:
					json_fallback_used = True
					json_mode_next = True
					continue
				error = ValueError(str(error))

			if isinstance(error, (ValidationError, ValueError, TypeError)):
				if corrections >= self.settings.MAX_CORRECTIONS:
					raise ToolFailure(
						ToolError(
							tool="llm",
							error_class=ErrorClass.SCHEMA,
							message=(
								"structured output invalid after "
								f"{corrections} corrections: {last_validation_error}"
							),
							attempt=attempt,
						)
					) from error
				corrections += 1
				current_user = (
					f"{user}\n\nCorrection {corrections}: your previous output failed "
					f"validation. Validation error: {error}. Return a corrected JSON object."
				)
				continue
			raise error


@dataclass(frozen=True)
class LLMCall:
	"""Captured inputs for a fake LLM request."""

	role: Role
	system: str
	user: str
	schema: type[BaseModel]


FakeResponse = object | Callable[[LLMCall], object]


class FakeLLMClient:
	"""Queue-backed fake LLM for tests and offline pipeline runs."""

	def __init__(self, responses: Mapping[str, Sequence[FakeResponse]] | None = None) -> None:
		self._responses: dict[str, deque[FakeResponse]] = defaultdict(deque)
		if responses is not None:
			for schema_name, queue in responses.items():
				self._responses[schema_name].extend(queue)
		self.calls: list[LLMCall] = []

	def enqueue(self, schema: type[BaseModel] | str, response: FakeResponse) -> None:
		schema_name = schema if isinstance(schema, str) else schema.__name__
		self._responses[schema_name].append(response)

	def structured(
		self,
		*,
		role: Role,
		system: str,
		user: str,
		schema: type[T],
		validate: Callable[[T], None] | None = None,
	) -> LLMResult[T]:
		del validate
		call = LLMCall(role=role, system=system, user=user, schema=schema)
		self.calls.append(call)
		queue = self._responses[schema.__name__]
		if not queue:
			raise RuntimeError(f"FakeLLMClient has no queued response for {schema.__name__}")
		response = queue.popleft()
		value = response(call) if callable(response) else response
		if isinstance(value, BaseException):
			raise value
		if isinstance(value, LLMResult):
			return value  # type: ignore[return-value]
		model = "fake"
		return LLMResult(
			parsed=value,  # type: ignore[arg-type]
			input_tokens=0,
			output_tokens=0,
			model=model,
			latency_ms=0.0,
		)
