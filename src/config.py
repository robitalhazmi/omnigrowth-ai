"""Global configuration for OmniGrowth-AI.

Centralizes filesystem paths, DuckDB memory guardrails, data-pipeline settings,
default model hyperparameters and GenAI resilience settings. All settings are
immutable ``dataclasses`` so they can be shared safely across threads.

The DuckDB guardrail (``192MB`` memory / ``1`` thread) is a hard ceiling: values
supplied through environment variables may only *lower* these limits, never raise
them, so the shared 1 vCPU / 2 GB production VPS is always protected.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Final

try:  # python-dotenv is optional at import time so src/ stays importable anywhere.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only without python-dotenv
    load_dotenv = None

logger = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

#: Hard guardrails mandated by AGENTS.md (shared 1 vCPU / 2 GB VPS).
MAX_DUCKDB_MEMORY_MB: Final[int] = 192
MAX_DUCKDB_THREADS: Final[int] = 1
MIN_DUCKDB_MEMORY_MB: Final[int] = 32

#: Global deterministic seed for every stochastic step in the offline pipeline.
RANDOM_SEED: Final[int] = 42

_MEMORY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(KB|MB|GB|KIB|MIB|GIB)?\s*$", re.IGNORECASE
)
_UNIT_TO_MB: Final[dict[str, float]] = {
    "KB": 1 / 1000,
    "MB": 1.0,
    "GB": 1000.0,
    "KIB": 1024 / 1_000_000,
    "MIB": 1_048_576 / 1_000_000,
    "GIB": 1_073_741_824 / 1_000_000,
}


def parse_memory_limit_mb(value: str) -> int:
    """Parses a human-readable memory string (e.g. ``'192MB'``, ``'1GB'``) into megabytes."""
    match = _MEMORY_PATTERN.match(value)
    if match is None:
        raise ValueError(f"Unparseable memory limit: {value!r}")
    amount = float(match.group(1))
    unit = (match.group(2) or "MB").upper()
    return int(amount * _UNIT_TO_MB[unit])


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout of the repository and all data artifacts."""

    root: Path = PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def fallback_dir(self) -> Path:
        return self.data_dir / "fallback"

    @property
    def looker_dir(self) -> Path:
        return self.root / "looker"

    @property
    def ga4_events(self) -> Path:
        return self.processed_dir / "ga4_events.parquet"

    @property
    def regional_demand(self) -> Path:
        return self.processed_dir / "regional_demand.parquet"

    @property
    def ab_experiment_logs(self) -> Path:
        return self.processed_dir / "ab_experiment_logs.parquet"

    @property
    def mql_funnel(self) -> Path:
        return self.processed_dir / "mql_funnel.parquet"

    @property
    def looker_mart_csv(self) -> Path:
        return self.processed_dir / "looker_executive_mart.csv"

    @property
    def lineage_json(self) -> Path:
        return self.processed_dir / "data_lineage.json"

    @property
    def forecasts_pretrained(self) -> Path:
        return self.processed_dir / "forecasts_pretrained.parquet"

    @property
    def scurve_params(self) -> Path:
        return self.processed_dir / "scurve_params.parquet"

    @property
    def model_backtest_metrics(self) -> Path:
        return self.processed_dir / "model_backtest_metrics.parquet"

    @property
    def genai_cache(self) -> Path:
        return self.fallback_dir / "genai_cache.json"

    def ensure_directories(self) -> None:
        """Creates every data directory if it does not already exist."""
        for directory in (self.raw_dir, self.processed_dir, self.fallback_dir):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DuckDBSettings:
    """DuckDB memory discipline settings (hard-capped at 192MB / 1 thread).

    Attributes:
        memory_limit_mb: DuckDB buffer-manager limit in megabytes.
        threads: Number of DuckDB worker threads.
    """

    memory_limit_mb: int = MAX_DUCKDB_MEMORY_MB
    threads: int = MAX_DUCKDB_THREADS

    def __post_init__(self) -> None:
        if not MIN_DUCKDB_MEMORY_MB <= self.memory_limit_mb <= MAX_DUCKDB_MEMORY_MB:
            raise ValueError(
                f"memory_limit_mb must be within [{MIN_DUCKDB_MEMORY_MB}, "
                f"{MAX_DUCKDB_MEMORY_MB}], got {self.memory_limit_mb}"
            )
        if not 1 <= self.threads <= MAX_DUCKDB_THREADS:
            raise ValueError(f"threads must be within [1, {MAX_DUCKDB_THREADS}], got {self.threads}")

    @property
    def memory_limit(self) -> str:
        """DuckDB-formatted memory limit string, e.g. ``'192MB'``."""
        return f"{self.memory_limit_mb}MB"

    def init_statements(self) -> tuple[str, ...]:
        """Returns the mandatory SQL statements executed on every new connection."""
        return (
            f"SET memory_limit = '{self.memory_limit}'",
            f"SET threads = {self.threads}",
        )

    @classmethod
    def from_env(cls) -> DuckDBSettings:
        """Builds settings from ``DUCKDB_MEMORY_LIMIT`` / ``DUCKDB_THREADS`` (clamped to caps)."""
        memory_mb = MAX_DUCKDB_MEMORY_MB
        raw_memory = os.getenv("DUCKDB_MEMORY_LIMIT")
        if raw_memory:
            try:
                memory_mb = parse_memory_limit_mb(raw_memory)
            except ValueError:
                logger.warning("Ignoring malformed DUCKDB_MEMORY_LIMIT=%r", raw_memory)
        memory_mb = max(MIN_DUCKDB_MEMORY_MB, min(memory_mb, MAX_DUCKDB_MEMORY_MB))

        threads = MAX_DUCKDB_THREADS
        raw_threads = os.getenv("DUCKDB_THREADS")
        if raw_threads:
            try:
                threads = int(raw_threads)
            except ValueError:
                logger.warning("Ignoring malformed DUCKDB_THREADS=%r", raw_threads)
        threads = max(1, min(threads, MAX_DUCKDB_THREADS))
        return cls(memory_limit_mb=memory_mb, threads=threads)


@dataclass(frozen=True)
class DataPipelineSettings:
    """Offline ETL settings for the public benchmark datasets."""

    demand_start_date: str = "2017-01-01"
    demand_end_date: str = "2018-08-31"
    ga4_start_date: str = "2020-11-01"
    ga4_end_date: str = "2021-01-31"
    ga4_n_users: int = 90_000
    ab_pre_experiment_days: int = 30
    parquet_compression: str = "zstd"
    processed_budget_mb: float = 25.0
    http_timeout_s: float = 60.0
    http_max_retries: int = 3


@dataclass(frozen=True)
class ProphetHyperparameters:
    """Default Meta Prophet hyperparameters for offline training."""

    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    holidays_prior_scale: float = 10.0
    seasonality_mode: str = "multiplicative"
    weekly_seasonality: bool = True
    yearly_seasonality: bool = True
    holiday_country: str = "BR"  # Olist demand is Brazilian; switch to "ID" for Indonesia.


@dataclass(frozen=True)
class ARIMAHyperparameters:
    """Default Nixtla StatsForecast AutoARIMA search space."""

    season_length: int = 7
    max_p: int = 5
    max_q: int = 5
    max_P: int = 2  # noqa: N815 - mirrors the statsforecast argument name
    max_Q: int = 2  # noqa: N815 - mirrors the statsforecast argument name
    information_criterion: str = "aicc"


@dataclass(frozen=True)
class ForecastSettings:
    """Forecast horizon, backtesting and online-serving limits."""

    horizon_days: int = 90
    cv_windows: int = 3
    cv_step_days: int = 30
    interval_levels: tuple[int, ...] = (80, 95)
    max_online_points: int = 730
    prophet: ProphetHyperparameters = field(default_factory=ProphetHyperparameters)
    arima: ARIMAHyperparameters = field(default_factory=ARIMAHyperparameters)


@dataclass(frozen=True)
class GenAISettings:
    """GenAI provider hierarchy and resilience settings."""

    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-2.5-flash"
    timeout_s: float = 10.0
    max_retries: int = 3
    backoff_base_s: float = 0.5
    groq_api_key: str | None = field(default=None, repr=False)
    gemini_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> GenAISettings:
        """Reads optional API keys; placeholders from ``.env.example`` count as unset."""

        def _key(name: str) -> str | None:
            value = (os.getenv(name) or "").strip()
            return None if not value or value.startswith("your_") else value

        return cls(groq_api_key=_key("GROQ_API_KEY"), gemini_api_key=_key("GEMINI_API_KEY"))


@dataclass(frozen=True)
class AppConfig:
    """Top-level immutable application configuration."""

    paths: ProjectPaths = field(default_factory=ProjectPaths)
    duckdb: DuckDBSettings = field(default_factory=DuckDBSettings)
    pipeline: DataPipelineSettings = field(default_factory=DataPipelineSettings)
    forecast: ForecastSettings = field(default_factory=ForecastSettings)
    genai: GenAISettings = field(default_factory=GenAISettings)
    looker_embed_url: str | None = None
    random_seed: int = RANDOM_SEED


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Returns the process-wide configuration singleton built from the environment."""
    embed_url = (os.getenv("LOOKER_STUDIO_EMBED_URL") or "").strip() or None
    if embed_url and "your-report-id" in embed_url:
        embed_url = None
    return AppConfig(
        duckdb=DuckDBSettings.from_env(),
        genai=GenAISettings.from_env(),
        looker_embed_url=embed_url,
    )
