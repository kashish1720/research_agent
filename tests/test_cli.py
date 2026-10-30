"""CLI JSON, files, trace output, and process status tests."""

from __future__ import annotations

import json

from research_agent.config import Settings
from research_agent.pipeline import ConfigError
from research_agent.schemas import (
    Confidence,
    ConfidenceReport,
    Coverage,
    FinalAnswer,
    RunStatus,
    UsageSummary,
)


def _answer(status: RunStatus = RunStatus.DONE) -> FinalAnswer:
    return FinalAnswer(
        run_id="cli-test",
        user_question="Research a topic",
        short_answer="Reliable evidence addresses the topic.",
        key_findings=[],
        sources_used=[],
        confidence=ConfidenceReport(
            level=Confidence.LOW,
            coverage_ratio=0,
            avg_source_quality=0,
            min_independent_domains=0,
            unresolved_conflicts=0,
            snippet_only_ratio=0,
            verification_complete=False,
            reasons=["No findings."],
        ),
        limitations_and_assumptions=[],
        suggested_next_steps=["Review relevant primary sources."],
        conflicts=[],
        coverage=Coverage(total=0, covered=0, per_subquestion=[]),
        status=status,
        reasons=[],
        usage=UsageSummary(
            llm_calls=0,
            input_tokens=0,
            output_tokens=0,
            searches=0,
            fetches=0,
            tool_retries=0,
            latency_s=0,
        ),
    )


def test_json_output_parses_as_final_answer_with_no_extra_stdout(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    import main as cli

    answer = _answer()
    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda: Settings(LOG_DIR=str(tmp_path / "logs")),
    )
    monkeypatch.setattr(cli, "run_pipeline", lambda *_args, **_kwargs: answer)

    assert cli.main(["Research a topic", "--json"]) == 0
    captured = capsys.readouterr()

    parsed = FinalAnswer.model_validate_json(captured.out)
    assert parsed == answer
    assert captured.err == ""


def test_cli_flags_save_json_and_disable_cache(tmp_path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    import main as cli

    answer = _answer()
    settings = Settings(LOG_DIR=str(tmp_path / "logs"))
    observed: list[Settings] = []

    def fake_pipeline(_question: str, *, settings: Settings) -> FinalAnswer:
        observed.append(settings)
        return answer

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "run_pipeline", fake_pipeline)
    save_path = tmp_path / "out" / "answer.json"

    result = cli.main(
        [
            "Research a topic",
            "--no-cache",
            "--faults",
            "timeout:0",
            "--save",
            str(save_path),
        ]
    )
    capsys.readouterr()

    assert result == 0
    assert observed[0].USE_CACHE is False
    assert observed[0].FAULT_INJECT == "timeout:0"
    assert FinalAnswer.model_validate_json(save_path.read_text(encoding="utf-8")) == answer


def test_cli_failed_status_returns_exit_code_two(tmp_path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    import main as cli

    monkeypatch.setattr(cli, "load_settings", lambda: Settings())
    monkeypatch.setattr(
        cli,
        "run_pipeline",
        lambda *_args, **_kwargs: _answer(RunStatus.FAILED_GRACEFULLY),
    )

    assert cli.main(["Research a topic", "--json"]) == 2
    assert FinalAnswer.model_validate_json(capsys.readouterr().out).status is (
        RunStatus.FAILED_GRACEFULLY
    )


def test_cli_config_error_returns_one(tmp_path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    import main as cli

    monkeypatch.setattr(cli, "load_settings", lambda: Settings())

    def fail(*_args, **_kwargs):
        raise ConfigError("missing configuration")

    monkeypatch.setattr(cli, "run_pipeline", fail)

    assert cli.main(["Research a topic", "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "configuration error" in captured.err


def test_json_trace_is_written_to_stderr_and_question_file_is_read(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    import main as cli

    answer = _answer()
    logs = tmp_path / "logs"
    settings = Settings(LOG_DIR=str(logs))
    logs.mkdir()
    (logs / "cli-test.jsonl").write_text(
        json.dumps(
            {
                "event": "note",
                "details": {
                    "trace_summary": {
                        "plan": {
                            "duration_ms": 1.0,
                            "tokens_in": 2,
                            "tokens_out": 1,
                            "tool_calls": 0,
                            "retries": 0,
                            "failures": 0,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    observed: list[str] = []

    def fake_pipeline(question: str, *, settings: Settings) -> FinalAnswer:
        observed.append(question)
        return answer

    question_file = tmp_path / "question.txt"
    question_file.write_text("Research a topic\n", encoding="utf-8")
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "run_pipeline", fake_pipeline)

    assert cli.main(["--question-file", str(question_file), "--json", "--trace"]) == 0
    captured = capsys.readouterr()

    assert FinalAnswer.model_validate_json(captured.out) == answer
    assert observed == ["Research a topic"]
    assert "plan | 1.0 | 2/1" in captured.err
