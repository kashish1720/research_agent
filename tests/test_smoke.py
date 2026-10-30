"""Smoke tests."""

import research_agent


def test_version() -> None:
    assert hasattr(research_agent, "__version__")
    assert isinstance(research_agent.__version__, str)
    assert research_agent.__version__ != ""

