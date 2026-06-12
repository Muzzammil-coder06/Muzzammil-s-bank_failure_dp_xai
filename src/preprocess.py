"""Reusable preprocessing: cleaning, outlier handling, scaling, and splitting."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config
from src.feature_engineering import engineer_features
from src.utils import setup_logging

logger = setup_logging()


class PreprocessError(Exception):
    """Raised when preprocessing or data validation fails."""


@dataclass
class PreprocessResult:
    """Container for preprocessed train/test splits and fitted transformers."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    metadata: pd.DataFrame


def _validate_input_schema(df: pd.DataFrame) -> None:
    """Validate that required identifier and target columns exist."""
    if df is None or df.empty:
        raise PreprocessError("Input dataset is empty.")

    required = set(config.ID_COLUMNS + [config.TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise PreprocessError(f"Dataset missing required columns: {sorted(missing)}")

    if df[config.TARGET_COLUMN].isna().all():
        raise PreprocessError("Target column contains only missing values.")

    if df[config.TARGET_COLUMN].nunique() < 2:
        raise PreprocessError("Target column must contain at least two classes.")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate bank-quarter observations."""
    before = len(df)
    deduped = df.drop_duplicates(subset=config.ID_COLUMNS, keep="last")
    removed = before - len(deduped)
    if removed:
        logger.info("Removed %d duplicate rows on %s", removed, config.ID_COLUMNS)
    return deduped


def drop_invalid_rows(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Remove rows with non-positive assets or invalid target values."""
    cleaned = df.copy()
    if "ASSET" in cleaned.columns:
        cleaned = cleaned[cleaned["ASSET"] > 0]

    cleaned = cleaned[cleaned[config.TARGET_COLUMN].isin([0, 1])]
    cleaned = cleaned.dropna(subset=[config.TARGET_COLUMN])

    if cleaned.empty:
        raise PreprocessError("No rows remain after invalid-row filtering.")

    available_features = [col for col in feature_columns if col in cleaned.columns]
    if not available_features:
        raise PreprocessError("No feature columns available after row filtering.")

    return cleaned


def cap_outliers_iqr(
    df: pd.DataFrame,
    feature_columns: list[str],
    multiplier: float = config.OUTLIER_IQR_MULTIPLIER,
) -> pd.DataFrame:
    """Winsorize numeric features at IQR-based bounds."""
    capped = df.copy()
    capped_count = 0

    for col in feature_columns:
        if col not in capped.columns or not np.issubdtype(capped[col].dtype, np.number):
            continue

        series = capped[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        before = series.copy()
        capped[col] = series.clip(lower=lower, upper=upper)
        capped_count += int((before != capped[col]).sum())

    logger.info("Outlier capping adjusted %d feature values (IQR multiplier=%.1f)", capped_count, multiplier)
    return capped


def _check_class_counts(y: pd.Series, context: str) -> None:
    """Ensure each class has enough samples for stratified modeling."""
    counts = y.value_counts()
    for label, count in counts.items():
        if count < config.MIN_SAMPLES_PER_CLASS:
            raise PreprocessError(
                f"{context}: class {label} has only {count} samples "
                f"(minimum required: {config.MIN_SAMPLES_PER_CLASS})."
            )


def split_features_target(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Separate feature matrix, target vector, and identifier metadata."""
    available_features = [col for col in feature_columns if col in df.columns]
    if not available_features:
        raise PreprocessError("No feature columns found for model training.")

    X = df[available_features].copy()
    y = df[config.TARGET_COLUMN].astype(int).copy()
    metadata = df[config.ID_COLUMNS].copy()
    return X, y, metadata


def fit_transform_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer, StandardScaler]:
    """
    Fit imputer and scaler on training data and transform both splits.

    Median imputation and standard scaling are fit exclusively on the training set
    to avoid data leakage.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imputed),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, imputer, scaler


def save_preprocess_artifacts(result: PreprocessResult) -> None:
    """Persist splits, transformers, and feature manifest."""
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    result.X_train.assign(**{config.TARGET_COLUMN: result.y_train}).to_csv(
        config.TRAIN_DATA_FILE, index=False
    )
    result.X_test.assign(**{config.TARGET_COLUMN: result.y_test}).to_csv(
        config.TEST_DATA_FILE, index=False
    )
    result.metadata.to_csv(config.SPLIT_METADATA_FILE, index=False)

    joblib.dump(result.imputer, config.IMPUTER_FILE)
    joblib.dump(result.scaler, config.SCALER_FILE)
    joblib.dump(result.feature_columns, config.FEATURE_COLUMNS_FILE)

    logger.info("Saved preprocessing artifacts to %s", config.PROCESSED_DATA_DIR.name)


def load_preprocess_artifacts() -> PreprocessResult:
    """Load previously saved preprocessing artifacts."""
    required_files = [
        config.TRAIN_DATA_FILE,
        config.TEST_DATA_FILE,
        config.IMPUTER_FILE,
        config.SCALER_FILE,
        config.FEATURE_COLUMNS_FILE,
    ]
    missing = [path for path in required_files if not path.exists()]
    if missing:
        raise PreprocessError(
            "Missing preprocessing artifacts: " + ", ".join(p.name for p in missing)
        )

    train_df = pd.read_csv(config.TRAIN_DATA_FILE)
    test_df = pd.read_csv(config.TEST_DATA_FILE)
    feature_columns: list[str] = joblib.load(config.FEATURE_COLUMNS_FILE)

    y_train = pd.to_numeric(train_df.pop(config.TARGET_COLUMN), errors="coerce")
    y_test = pd.to_numeric(test_df.pop(config.TARGET_COLUMN), errors="coerce")

    if y_train.isna().any() or y_test.isna().any():
        raise PreprocessError("Target column contains missing values in saved train/test files.")

    return PreprocessResult(
        X_train=train_df[feature_columns],
        X_test=test_df[feature_columns],
        y_train=y_train.astype(int),
        y_test=y_test.astype(int),
        feature_columns=feature_columns,
        imputer=joblib.load(config.IMPUTER_FILE),
        scaler=joblib.load(config.SCALER_FILE),
        metadata=pd.read_csv(config.SPLIT_METADATA_FILE),
    )


def preprocess_pipeline(df: pd.DataFrame, *, save: bool = True) -> PreprocessResult:
    """
    Run the full preprocessing pipeline on a labeled dataset.

    Steps: validate → deduplicate → feature engineering → filter invalid rows
    → outlier capping → stratified split → impute → scale → persist artifacts.
    """
    _validate_input_schema(df)
    logger.info("Starting preprocessing on %d rows", len(df))

    deduped = remove_duplicates(df)
    featured, feature_columns = engineer_features(deduped)
    feature_columns = list(dict.fromkeys(feature_columns))
    cleaned = drop_invalid_rows(featured, feature_columns)
    capped = cap_outliers_iqr(cleaned, feature_columns)

    X, y, metadata = split_features_target(capped, feature_columns)
    _check_class_counts(y, context="Full dataset")

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        metadata,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )

    _check_class_counts(y_train, context="Training split")
    _check_class_counts(y_test, context="Test split")

    X_train_scaled, X_test_scaled, imputer, scaler = fit_transform_train_test(X_train, X_test)

    X_train_scaled = X_train_scaled.reset_index(drop=True)
    X_test_scaled = X_test_scaled.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    meta_train = meta_train.reset_index(drop=True)
    meta_test = meta_test.reset_index(drop=True)

    result = PreprocessResult(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        feature_columns=feature_columns,
        imputer=imputer,
        scaler=scaler,
        metadata=pd.concat([meta_train, meta_test], ignore_index=True),
    )

    logger.info(
        "Preprocessing complete: train=%s, test=%s, positives(train)=%d, positives(test)=%d",
        result.X_train.shape,
        result.X_test.shape,
        int(result.y_train.sum()),
        int(result.y_test.sum()),
    )

    if save:
        save_preprocess_artifacts(result)

    return result


if __name__ == "__main__":
    from src.data_loader import load_dataset, validate_dataset
    from src.utils import ensure_directories, set_random_seeds

    ensure_directories()
    set_random_seeds()
    dataset = load_dataset()
    validate_dataset(dataset)
    output = preprocess_pipeline(dataset)
    print(
        f"Train: {output.X_train.shape}, Test: {output.X_test.shape}, "
        f"Features: {len(output.feature_columns)}"
    )
