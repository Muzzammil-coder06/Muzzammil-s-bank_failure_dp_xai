"""Schema-aware financial ratio and indicator feature engineering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.utils import setup_logging

logger = setup_logging()


class FeatureEngineeringError(Exception):
    """Raised when feature engineering cannot produce a valid feature set."""


@dataclass(frozen=True)
class FeatureSpec:
    """Definition of a derived feature and its required source columns."""

    name: str
    required_columns: tuple[str, ...]
    builder: Callable[[pd.DataFrame], pd.Series]


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    min_denominator: float = 1.0,
) -> pd.Series:
    """Compute a ratio with protection against division by zero."""
    denom = denominator.replace(0, np.nan)
    denom = denom.where(denom.abs() >= min_denominator, np.nan)
    return numerator / denom


def _build_feature_specs() -> list[FeatureSpec]:
    """Return all candidate derived features supported by FDIC financial fields."""
    return [
        FeatureSpec(
            name="capital_adequacy_ratio",
            required_columns=("EQ", "ASSET"),
            builder=lambda df: _safe_ratio(df["EQ"], df["ASSET"]),
        ),
        FeatureSpec(
            name="tier1_capital_ratio",
            required_columns=("RBCT1", "ASSET"),
            builder=lambda df: _safe_ratio(df["RBCT1"], df["ASSET"]),
        ),
        FeatureSpec(
            name="liquidity_ratio",
            required_columns=("DEP", "ASSET"),
            builder=lambda df: _safe_ratio(df["DEP"], df["ASSET"]),
        ),
        FeatureSpec(
            name="domestic_deposit_ratio",
            required_columns=("DEPDOM", "ASSET"),
            builder=lambda df: _safe_ratio(df["DEPDOM"], df["ASSET"]),
        ),
        FeatureSpec(
            name="debt_ratio",
            required_columns=("LIAB", "ASSET"),
            builder=lambda df: _safe_ratio(df["LIAB"], df["ASSET"]),
        ),
        FeatureSpec(
            name="leverage_ratio",
            required_columns=("LIAB", "EQ"),
            builder=lambda df: _safe_ratio(df["LIAB"], df["EQ"]),
        ),
        FeatureSpec(
            name="loan_ratio",
            required_columns=("LNLSNET", "ASSET"),
            builder=lambda df: _safe_ratio(df["LNLSNET"], df["ASSET"]),
        ),
        FeatureSpec(
            name="gross_loan_ratio",
            required_columns=("LNLSGR", "ASSET"),
            builder=lambda df: _safe_ratio(df["LNLSGR"], df["ASSET"]),
        ),
        FeatureSpec(
            name="profitability_ratio",
            required_columns=("NETINC", "ASSET"),
            builder=lambda df: _safe_ratio(df["NETINC"], df["ASSET"]),
        ),
        FeatureSpec(
            name="interest_income_ratio",
            required_columns=("INTINC", "ASSET"),
            builder=lambda df: _safe_ratio(df["INTINC"], df["ASSET"]),
        ),
        FeatureSpec(
            name="noncurrent_loan_ratio",
            required_columns=("NCLNLS", "LNLSNET"),
            builder=lambda df: _safe_ratio(df["NCLNLS"], df["LNLSNET"]),
        ),
        FeatureSpec(
            name="loan_loss_reserve_ratio",
            required_columns=("LNATRES", "LNLSNET"),
            builder=lambda df: _safe_ratio(df["LNATRES"], df["LNLSNET"]),
        ),
        FeatureSpec(
            name="ore_to_assets",
            required_columns=("ORE", "ASSET"),
            builder=lambda df: _safe_ratio(df["ORE"], df["ASSET"]),
        ),
        FeatureSpec(
            name="equity_multiplier",
            required_columns=("ASSET", "EQ"),
            builder=lambda df: _safe_ratio(df["ASSET"], df["EQ"]),
        ),
    ]


def get_base_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Select numeric base columns suitable for modeling.

    Excludes identifiers, target, and non-numeric API metadata.
    """
    exclude = set(config.ID_COLUMNS + [config.TARGET_COLUMN, "ID", "FAILDATE", "NAME", "CITYST"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return sorted(col for col in numeric_cols if col not in exclude)


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add derived financial ratios and return the enriched frame with feature names.

    Features are created only when all required source columns are present.
    Infinite values from division are replaced with NaN.
    """
    if df is None or df.empty:
        raise FeatureEngineeringError("Cannot engineer features on an empty dataset.")

    enriched = df.copy()
    created_features: list[str] = []

    for spec in _build_feature_specs():
        if not all(col in enriched.columns for col in spec.required_columns):
            logger.debug("Skipping feature '%s': missing columns %s", spec.name, spec.required_columns)
            continue

        try:
            series = spec.builder(enriched).replace([np.inf, -np.inf], np.nan)
            enriched[spec.name] = series
            created_features.append(spec.name)
            logger.debug("Created feature '%s'", spec.name)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to create feature '%s': %s", spec.name, exc)

    base_features = get_base_feature_columns(df)
    all_features = base_features + [name for name in created_features if name not in base_features]

    if not all_features:
        raise FeatureEngineeringError("No modeling features available after feature engineering.")

    logger.info(
        "Feature engineering complete: %d base features, %d derived features",
        len(base_features),
        len(created_features),
    )
    return enriched, all_features


def summarize_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Return descriptive statistics for engineered and base features."""
    available = [col for col in feature_columns if col in df.columns]
    if not available:
        raise FeatureEngineeringError("No requested feature columns found in dataframe.")
    return df[available].describe().T


if __name__ == "__main__":
    from src.data_loader import load_dataset, validate_dataset
    from src.utils import ensure_directories, set_random_seeds

    ensure_directories()
    set_random_seeds()
    raw = load_dataset()
    validate_dataset(raw)
    featured, feature_names = engineer_features(raw)
    print(f"Engineered {len(feature_names)} features on {featured.shape[0]} rows")
    print("Derived features:", [c for c in feature_names if c not in get_base_feature_columns(raw)])
