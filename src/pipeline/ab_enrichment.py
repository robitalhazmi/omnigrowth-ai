"""CUPED covariate enrichment for the public landing-page A/B test.

The public ``ab_data.csv`` experiment (294k users, Jan 2017) records the real
randomized assignment and the real binary conversion outcome, but it carries no
pre-experiment behaviour, revenue or retention. CUPED and retention analyses need
those fields, so this module *imputes* them deterministically from a seeded
latent customer-value model that is conditioned on the real outcome:

* ``user_id``, ``experiment_group``, ``assigned_date`` and ``converted`` are real.
* ``device_os``, ``region_tier``, ``pre_exp_sessions``, ``pre_exp_spend``,
  ``post_exp_revenue``, ``retention_d7`` and ``retention_d30`` are simulated with a
  documented, configurable treatment effect (see :class:`ABEnrichmentConfig`).

The shared latent value drives both pre- and post-period spend, giving a
realistic ``Corr(pre_exp_spend, post_exp_revenue)`` of roughly 0.5-0.7.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

AB_COLUMNS: tuple[str, ...] = (
    "user_id", "experiment_group", "assigned_date", "device_os", "region_tier",
    "pre_exp_sessions", "pre_exp_spend", "converted", "post_exp_revenue",
    "retention_d7", "retention_d30",
)
_DEVICE_OS = (("Android", 0.45), ("iOS", 0.30), ("Windows", 0.17), ("macOS", 0.08))
_REGION_TIER = (("Tier 1 Metro", 0.40, 0.25), ("Tier 2 City", 0.35, 0.0), ("Tier 3 Rural", 0.25, -0.30))


@dataclass(frozen=True)
class ABEnrichmentConfig:
    """Parameters of the covariate-imputation model.

    Attributes:
        seed: RNG seed (deterministic output for a given input order).
        treatment_revenue_lift: Multiplicative lift on post-period spend in treatment.
        treatment_retention_logit: Additive logit lift on D7/D30 retention in treatment.
        conversion_value_shift: Latent-value shift for users who really converted.
    """

    seed: int = 42
    treatment_revenue_lift: float = 0.04
    treatment_retention_logit: float = 0.08
    conversion_value_shift: float = 0.35

    def as_dict(self) -> dict[str, float]:
        """Returns the configuration as a JSON-serializable dict."""
        return asdict(self)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def enrich_ab_logs(clean: pd.DataFrame, cfg: ABEnrichmentConfig | None = None) -> pd.DataFrame:
    """Adds CUPED covariates, revenue and retention to cleaned A/B assignments.

    Args:
        clean: One row per user with columns ``user_id``, ``timestamp``,
            ``group`` (``control``/``treatment``) and ``converted`` (0/1).
        cfg: Imputation parameters.

    Returns:
        DataFrame with the columns in :data:`AB_COLUMNS`, sorted by ``user_id``.

    Raises:
        ValueError: If required columns are missing or ``user_id`` is not unique.
    """
    cfg = cfg or ABEnrichmentConfig()
    required = {"user_id", "timestamp", "group", "converted"}
    if missing := required - set(clean.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    df = clean.sort_values("user_id", kind="stable").reset_index(drop=True)
    if df["user_id"].duplicated().any():
        raise ValueError("user_id must be unique (deduplicate before enrichment)")

    rng = np.random.default_rng(cfg.seed)
    n = len(df)
    converted = df["converted"].to_numpy(dtype=np.int8)
    treat = (df["group"].to_numpy() == "treatment").astype(float)

    os_idx = rng.choice(len(_DEVICE_OS), n, p=[w for _, w in _DEVICE_OS])
    tier_idx = rng.choice(len(_REGION_TIER), n, p=[w for _, w, _ in _REGION_TIER])
    tier_effect = np.asarray([e for _, _, e in _REGION_TIER])[tier_idx]

    z = rng.standard_normal(n) + cfg.conversion_value_shift * converted + tier_effect
    level = np.exp(3.3 + 0.55 * z)  # persistent monthly spend level (USD)

    pre_sessions = rng.poisson(np.exp(1.1 + 0.45 * z))
    pre_active = rng.random(n) < _sigmoid(-0.2 + 1.6 * z)
    pre_spend = np.where(pre_active, level * rng.lognormal(0.0, 0.45, n), 0.0)

    post_active = rng.random(n) < _sigmoid(-0.2 + 1.6 * z)
    post_base = np.where(post_active, level * rng.lognormal(0.0, 0.45, n), 0.0)
    order_value = converted * rng.lognormal(np.log(45.0) + 0.25 * z, 0.5, n)
    post_revenue = (post_base + order_value) * (1.0 + cfg.treatment_revenue_lift * treat)

    lift = cfg.treatment_retention_logit * treat
    d7 = rng.random(n) < _sigmoid(-1.3 + 0.6 * z + 1.1 * converted + lift)
    d30 = d7 & (rng.random(n) < _sigmoid(-0.3 + 0.5 * z + 0.6 * converted + lift))

    return pd.DataFrame(
        {
            "user_id": df["user_id"].to_numpy(dtype=np.int64),
            "experiment_group": df["group"].astype(str).to_numpy(),
            "assigned_date": pd.to_datetime(df["timestamp"]).dt.date.to_numpy(),
            "device_os": np.asarray([o for o, _ in _DEVICE_OS])[os_idx],
            "region_tier": np.asarray([t for t, _, _ in _REGION_TIER])[tier_idx],
            "pre_exp_sessions": pre_sessions.astype(np.int16),
            "pre_exp_spend": np.round(pre_spend, 2),
            "converted": converted,
            "post_exp_revenue": np.round(post_revenue, 2),
            "retention_d7": d7.astype(np.int8),
            "retention_d30": d30.astype(np.int8),
        },
        columns=list(AB_COLUMNS),
    )
