"""Tests for src/config.py guardrails and path layout."""

from __future__ import annotations

import pytest

from src.config import (
    MAX_DUCKDB_MEMORY_MB,
    MAX_DUCKDB_THREADS,
    PROJECT_ROOT,
    AppConfig,
    DuckDBSettings,
    GenAISettings,
    get_config,
    parse_memory_limit_mb,
)


def test_duckdb_defaults_match_agents_guardrails() -> None:
    settings = DuckDBSettings()
    assert settings.memory_limit == "192MB"
    assert settings.threads == 1
    assert settings.init_statements() == ("SET memory_limit = '192MB'", "SET threads = 1")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("192MB", 192), ("128 mb", 128), ("1GB", 1000), ("64", 64), ("100MiB", 104)],
)
def test_parse_memory_limit(raw: str, expected: int) -> None:
    assert parse_memory_limit_mb(raw) == expected


def test_parse_memory_limit_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_memory_limit_mb("lots")


def test_settings_reject_values_above_guardrail() -> None:
    with pytest.raises(ValueError):
        DuckDBSettings(memory_limit_mb=512)
    with pytest.raises(ValueError):
        DuckDBSettings(threads=4)


def test_env_can_lower_but_never_raise_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUCKDB_MEMORY_LIMIT", "2GB")
    monkeypatch.setenv("DUCKDB_THREADS", "8")
    clamped = DuckDBSettings.from_env()
    assert clamped.memory_limit_mb == MAX_DUCKDB_MEMORY_MB
    assert clamped.threads == MAX_DUCKDB_THREADS

    monkeypatch.setenv("DUCKDB_MEMORY_LIMIT", "128MB")
    assert DuckDBSettings.from_env().memory_limit_mb == 128

    monkeypatch.setenv("DUCKDB_MEMORY_LIMIT", "not-a-size")
    monkeypatch.setenv("DUCKDB_THREADS", "x")
    fallback = DuckDBSettings.from_env()
    assert (fallback.memory_limit_mb, fallback.threads) == (MAX_DUCKDB_MEMORY_MB, 1)


def test_placeholder_api_keys_are_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "your_free_groq_api_key_here")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    genai = GenAISettings.from_env()
    assert genai.groq_api_key is None and genai.gemini_api_key is None
    assert genai.timeout_s == 10.0

    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret_value")
    keyed = GenAISettings.from_env()
    assert keyed.groq_api_key == "gsk_secret_value"
    assert "gsk_secret_value" not in repr(keyed)  # secrets never leak into logs


def test_paths_and_singleton() -> None:
    cfg = get_config()
    assert isinstance(cfg, AppConfig)
    assert get_config() is cfg
    assert cfg.paths.root == PROJECT_ROOT
    assert cfg.paths.ga4_events.name == "ga4_events.parquet"
    assert cfg.paths.processed_dir.parent == cfg.paths.data_dir
    assert cfg.forecast.max_online_points <= 730
    assert cfg.forecast.interval_levels == (80, 95)
