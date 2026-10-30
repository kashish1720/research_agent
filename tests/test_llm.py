"""Tests for the fake and resilient structured LLM clients."""

from __future__ import annotations

from types import SimpleNamespace

import httpx2
import openai
import pytest
from pydantic import SecretStr

from research_agent.budget import Budget
from research_agent.config import Settings
from research_agent.errors import ToolFailure
from research_agent.llm import (
    FakeLLMClient,
    GenerationRetryRequired,
    GroqClient,
    LLMResult,
    ResilientLLM,
    _schema_response_format,
)
from research_agent.schemas import ErrorClass, Plan, ToolError
from research_agent.tracing import Tracer


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "MAX_RETRIES": 2,
        "MAX_CORRECTIONS": 2,
        "BACKOFF_BASE_S": 0.1,
        "MAX_LLM_CALLS": 10,
        "MAX_TOTAL_TOKENS": 1_000,
    }
    values.update(overrides)
    return Settings(**values)


def _valid_plan() -> Plan:
    return Plan.model_validate(
        {
            "restated_question": "A research question",
            "is_answerable": True,
            "assumptions": [],
            "sub_questions": [
                {
                    "id": f"q{index}",
                    "text": f"Independent aspect {index}",
                    "queries": [f"example topic aspect {index}"],
                    "strategy": "search_and_fetch",
                    "time_sensitive": False,
                    "rationale": "Need a specific source",
                }
                for index in (1, 2)
            ],
        }
    )


def _runtime(tmp_path, settings: Settings | None = None):  # type: ignore[no-untyped-def]
    active_settings = settings or _settings()
    active_settings.CACHE_DIR = str(tmp_path / "cache")
    return (
        Budget(active_settings),
        Tracer("llm-test", tmp_path),
        active_settings,
    )


def _transient() -> ToolFailure:
    return ToolFailure(
        ToolError(tool="llm", error_class=ErrorClass.TRANSIENT, message="temporary")
    )


def test_transient_retry_then_success_consumes_and_traces_each_attempt(tmp_path) -> None:  # type: ignore[no-untyped-def]
    budget, tracer, settings = _runtime(tmp_path)
    fake = FakeLLMClient({"Plan": [lambda _call: _transient(), _valid_plan()]})
    delays: list[float] = []
    llm = ResilientLLM(fake, settings, budget, tracer, sleep_fn=delays.append)

    result = llm.structured(role="cheap", system="system", user="question", schema=Plan)

    assert result.parsed == _valid_plan()
    assert len(fake.calls) == 2
    assert budget.llm_calls == 2
    assert budget.tool_retries == 1
    assert delays[0] >= settings.BACKOFF_BASE_S
    assert [event["event"] for event in tracer.events].count("llm_call") == 2


def test_rate_limit_retry_after_is_honored(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    budget, tracer, settings = _runtime(tmp_path)
    rate_limited = _transient()
    rate_limited.retry_after_s = 3.5
    fake = FakeLLMClient({"Plan": [lambda _call: rate_limited, _valid_plan()]})
    monkeypatch.setattr("research_agent.llm.random.uniform", lambda _low, _high: 0.0)
    delays: list[float] = []

    ResilientLLM(fake, settings, budget, tracer, sleep_fn=delays.append).structured(
        role="cheap", system="system", user="question", schema=Plan
    )

    assert delays == [3.5]


def test_invalid_output_is_corrected_with_distinct_prompt_and_validation_error(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    budget, tracer, settings = _runtime(tmp_path)
    fake = FakeLLMClient({"Plan": [{"not": "a plan"}, _valid_plan()]})
    llm = ResilientLLM(fake, settings, budget, tracer, sleep_fn=lambda _delay: None)

    llm.structured(role="cheap", system="system", user="original prompt", schema=Plan)

    assert len(fake.calls) == 2
    assert fake.calls[0].user == "original prompt"
    assert fake.calls[1].user != fake.calls[0].user
    assert "Validation error" in fake.calls[1].user
    assert "restated_question" in fake.calls[1].user
    assert budget.llm_calls == 2


def test_custom_validate_failure_triggers_correction(tmp_path) -> None:  # type: ignore[no-untyped-def]
    budget, tracer, settings = _runtime(tmp_path)
    fake = FakeLLMClient({"Plan": [_valid_plan(), _valid_plan()]})
    calls = 0

    def validate(_plan: Plan) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("custom check failed")

    llm = ResilientLLM(fake, settings, budget, tracer, sleep_fn=lambda _delay: None)
    llm.structured(
        role="cheap",
        system="system",
        user="question",
        schema=Plan,
        validate=validate,
    )

    assert calls == 2
    assert "custom check failed" in fake.calls[1].user
    assert budget.llm_calls == 2


def test_correction_limit_maps_to_schema_failure(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(MAX_CORRECTIONS=2)
    budget, tracer, _settings_result = _runtime(tmp_path, settings)
    fake = FakeLLMClient({"Plan": [{}, {}, {}]})
    llm = ResilientLLM(fake, settings, budget, tracer, sleep_fn=lambda _delay: None)

    with pytest.raises(ToolFailure) as exc_info:
        llm.structured(role="cheap", system="system", user="question", schema=Plan)

    assert exc_info.value.error.error_class is ErrorClass.SCHEMA
    assert len(fake.calls) == 3
    assert budget.llm_calls == 3
    assert len({call.user for call in fake.calls}) == 3


def test_validation_error_is_traced_and_included_in_final_failure(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(MAX_CORRECTIONS=1)
    budget, tracer, _settings_result = _runtime(tmp_path, settings)
    fake = FakeLLMClient({"Plan": [{}, {}]})
    llm = ResilientLLM(fake, settings, budget, tracer, sleep_fn=lambda _delay: None)

    with pytest.raises(ToolFailure, match="restated_question") as exc_info:
        llm.structured(role="cheap", system="system", user="question", schema=Plan)

    failed_calls = [event for event in tracer.events if event["event"] == "llm_call"]
    assert len(failed_calls) == 2
    assert all(
        "restated_question" in event["details"]["validation_error"]
        for event in failed_calls
    )
    assert all(len(event["details"]["validation_error"]) <= 300 for event in failed_calls)
    assert "restated_question" in exc_info.value.error.message


def test_budget_exhaustion_happens_before_an_attempt(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(MAX_LLM_CALLS=1)
    budget, tracer, _settings_result = _runtime(tmp_path, settings)
    fake = FakeLLMClient({"Plan": [_valid_plan(), _valid_plan()]})
    llm = ResilientLLM(fake, settings, budget, tracer, sleep_fn=lambda _delay: None)
    llm.structured(role="cheap", system="system", user="first", schema=Plan)

    with pytest.raises(Exception, match="llm_calls"):
        llm.structured(role="cheap", system="system", user="second", schema=Plan)

    assert budget.llm_calls == 1
    assert len(fake.calls) == 1


def test_fake_queue_empty_has_clear_error() -> None:
    fake = FakeLLMClient()

    with pytest.raises(RuntimeError, match="no queued response for Plan"):
        fake.structured(role="cheap", system="s", user="u", schema=Plan)


def test_groq_model_defaults_and_strict_schema_without_pydantic_constraints() -> None:
    settings = _settings()
    groq = GroqClient(settings, client=object())
    response_format = _schema_response_format(Plan)
    schema_json = response_format["json_schema"]

    assert groq.model_for("cheap") == "openai/gpt-oss-20b"
    assert groq.model_for("strong") == "openai/gpt-oss-120b"
    assert response_format["type"] == "json_schema"
    assert schema_json["strict"] is True
    assert "minItems" not in str(schema_json["schema"])


def test_fake_preserves_explicit_usage_result() -> None:
    expected = LLMResult(
        parsed=_valid_plan(),
        input_tokens=12,
        output_tokens=7,
        model="fake-model",
        latency_ms=3.0,
    )
    fake = FakeLLMClient({"Plan": [expected]})

    actual = fake.structured(role="cheap", system="s", user="u", schema=Plan)

    assert actual == expected


class ChatCompletionStub:
    def __init__(self, *, reject_schema: bool = False) -> None:
        self.reject_schema = reject_schema
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        response_format = kwargs["response_format"]
        if self.reject_schema and response_format["type"] == "json_schema":  # type: ignore[index]
            request = httpx2.Request(
                "POST", "https://api.groq.com/openai/v1/chat/completions"
            )
            response = httpx2.Response(400, request=request)
            raise openai.BadRequestError(
                "response_format json_schema is unsupported",
                response=response,
                body={"error": "unsupported format"},
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=_valid_plan().model_dump_json(),
                        refusal=None,
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=21, completion_tokens=33),
            model=str(kwargs["model"]),
        )


def test_groq_structured_uses_strict_schema_low_reasoning_and_usage() -> None:
    settings = _settings(LLM_API_KEY=SecretStr("test-key"))
    completions = ChatCompletionStub()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    groq = GroqClient(settings, client=client)

    result = groq.structured(
        role="cheap",
        system="system",
        user="user",
        schema=Plan,
    )

    request = completions.calls[0]
    assert request["model"] == "openai/gpt-oss-20b"
    assert request["temperature"] == 0
    assert request["reasoning_effort"] == "low"
    assert request["max_completion_tokens"] >= 4_000
    assert request["timeout"] == settings.LLM_TIMEOUT_S
    assert request["response_format"]["json_schema"]["strict"] is True  # type: ignore[index]
    assert result.input_tokens == 21
    assert result.output_tokens == 33
    assert isinstance(result.parsed, Plan)


def test_groq_bad_schema_request_falls_back_to_json_mode(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(LLM_API_KEY=SecretStr("test-key"))
    settings.CACHE_DIR = str(tmp_path / "cache")
    completions = ChatCompletionStub(reject_schema=True)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    groq = GroqClient(settings, client=client)
    budget = Budget(settings)
    tracer = Tracer("groq-fallback-test", tmp_path)

    result = ResilientLLM(groq, settings, budget, tracer, sleep_fn=lambda _delay: None).structured(
        role="strong", system="system", user="user", schema=Plan
    )

    assert len(completions.calls) == 2
    assert completions.calls[0]["response_format"]["type"] == "json_schema"  # type: ignore[index]
    assert completions.calls[1]["response_format"] == {"type": "json_object"}
    assert completions.calls[1]["messages"][0]["content"] != "system"  # type: ignore[index]
    assert result.model == "openai/gpt-oss-120b"


def test_failed_generation_empty_json_fallback_enters_bounded_correction(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(MAX_CORRECTIONS=1)
    settings.CACHE_DIR = str(tmp_path / "cache")

    class GenerationFailureStub:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                request = httpx2.Request(
                    "POST", "https://api.groq.com/openai/v1/chat/completions"
                )
                response = httpx2.Response(400, request=request)
                raise openai.BadRequestError(
                    "Generated JSON does not match the expected schema",
                    response=response,
                    body={
                        "error": {
                            "failed_generation": '{"restated_question":"x", "is_answerable":true}'
                        }
                    },
                )
            content = "" if len(self.calls) == 2 else "not json"
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content, refusal=None))],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=1),
                model="openai/gpt-oss-20b",
            )

    completions = GenerationFailureStub()
    groq = GroqClient(
        settings,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    budget = Budget(settings)
    tracer = Tracer("empty-json-fallback-test", tmp_path)
    resilient = ResilientLLM(groq, settings, budget, tracer, sleep_fn=lambda _delay: None)

    with pytest.raises(ToolFailure) as exc_info:
        resilient.structured(role="cheap", system="system", user="original", schema=Plan)

    assert exc_info.value.error.error_class is ErrorClass.SCHEMA
    assert len(completions.calls) == 3
    assert [call["response_format"]["type"] for call in completions.calls] == [
        "json_schema",
        "json_object",
        "json_schema",
    ]  # type: ignore[index]
    assert completions.calls[1]["messages"][0]["content"] != "system"  # type: ignore[index]
    assert completions.calls[2]["messages"][1]["content"] != "original"  # type: ignore[index]
    assert budget.llm_calls == 3
    assert budget.input_tokens == 10
    assert budget.output_tokens == 2
    assert [event["event"] for event in tracer.events].count("llm_call") == 3


def test_valid_failed_generation_is_salvaged_without_second_request() -> None:
    class SalvageStub:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_kwargs: object) -> object:
            self.calls += 1
            request = httpx2.Request(
                "POST", "https://api.groq.com/openai/v1/chat/completions"
            )
            response = httpx2.Response(400, request=request)
            raise openai.BadRequestError(
                "Generated JSON does not match the expected schema",
                response=response,
                body={"error": {"failed_generation": _valid_plan().model_dump_json()}},
            )

    client = SalvageStub()
    groq = GroqClient(
        _settings(LLM_API_KEY=SecretStr("test-key")),
        client=SimpleNamespace(chat=SimpleNamespace(completions=client)),
    )

    result = groq.structured(role="cheap", system="system", user="user", schema=Plan)

    assert result.parsed == _valid_plan()
    assert client.calls == 1


def test_json_mode_schema_400_falls_through_to_bounded_correction(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(MAX_CORRECTIONS=1)
    settings.CACHE_DIR = str(tmp_path / "cache")

    class RepeatedGenerationFailureStub:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            if len(self.calls) in {1, 2}:
                request = httpx2.Request(
                    "POST", "https://api.groq.com/openai/v1/chat/completions"
                )
                response = httpx2.Response(400, request=request)
                raise openai.BadRequestError(
                    "Generated JSON does not match the expected schema",
                    response=response,
                    body={"error": {"failed_generation": ""}},
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}", refusal=None))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                model="openai/gpt-oss-20b",
            )

    completions = RepeatedGenerationFailureStub()
    groq = GroqClient(
        settings,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    resilient = ResilientLLM(
        groq,
        settings,
        Budget(settings),
        Tracer("one-json-fallback-test", tmp_path),
        sleep_fn=lambda _delay: None,
    )

    with pytest.raises(ToolFailure) as exc_info:
        resilient.structured(role="cheap", system="system", user="original", schema=Plan)

    assert exc_info.value.error.error_class is ErrorClass.SCHEMA
    assert [call["response_format"]["type"] for call in completions.calls] == [
        "json_schema",
        "json_object",
        "json_schema",
    ]  # type: ignore[index]


def test_debug_bad_request_logs_safe_required_fields(monkeypatch, caplog) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LLM_DEBUG", "1")
    class ErrorStub:
        def create(self, **_kwargs: object) -> object:
            request = httpx2.Request(
                "POST", "https://api.groq.com/openai/v1/chat/completions"
            )
            response = httpx2.Response(400, request=request)
            failed = (
                '{"failed_generation":"Bearer leaked-secret api_key=sk-secret-value '
                'and \\"token\\": \\"json-token-secret\\""}'
            )
            raise openai.BadRequestError(
                "Generated JSON does not match the expected schema",
                response=response,
                body={"error": {"failed_generation": failed}},
            )

    groq = GroqClient(
        _settings(LLM_API_KEY=SecretStr("never-log-this")),
        client=SimpleNamespace(chat=SimpleNamespace(completions=ErrorStub())),
    )
    with caplog.at_level("WARNING", logger="research_agent.llm"):
        with pytest.raises(GenerationRetryRequired):
            groq.structured(role="cheap", system="system", user="user", schema=Plan)

    diagnostic = caplog.text
    assert "finish_reason=None" in diagnostic
    assert "usage=None" in diagnostic
    assert "max_completion_tokens=8192" in diagnostic
    assert "failed_generation=" in diagnostic
    assert "leaked-secret" not in diagnostic
    assert "sk-secret-value" not in diagnostic
    assert "json-token-secret" not in diagnostic
    assert "never-log-this" not in diagnostic