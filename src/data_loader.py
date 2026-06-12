"""FDIC dataset acquisition and label construction."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src import config
from src.utils import setup_logging

logger = setup_logging()


class DataLoaderError(Exception):
    """Raised when dataset loading or validation fails."""


def _api_request(url: str) -> dict[str, Any]:
    """Execute an FDIC API GET request with retries."""
    last_error: Exception | None = None
    for attempt in range(1, config.FDIC_API_RETRY_ATTEMPTS + 1):
        try:
            request = Request(
                url,
                headers={"Accept": "application/json", "User-Agent": "bank-failure-dp-xai/1.0"},
            )
            with urlopen(request, timeout=config.FDIC_API_TIMEOUT_SECONDS) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning("API request failed (attempt %d/%d): %s", attempt, config.FDIC_API_RETRY_ATTEMPTS, exc)
            if attempt < config.FDIC_API_RETRY_ATTEMPTS:
                time.sleep(config.FDIC_API_RETRY_DELAY_SECONDS * attempt)
    raise DataLoaderError(f"FDIC API request failed after retries: {last_error}") from last_error


def _paginated_fetch(endpoint: str, params: dict[str, str]) -> list[dict[str, Any]]:
    """Fetch all pages from a paginated FDIC API endpoint."""
    records: list[dict[str, Any]] = []
    offset = 0

    while True:
        page_params = {**params, "limit": str(config.FDIC_API_PAGE_LIMIT), "offset": str(offset), "format": "json"}
        url = f"{endpoint}?{urlencode(page_params)}"
        payload = _api_request(url)
        page_data = payload.get("data", [])
        if not page_data:
            break

        for item in page_data:
            record = item.get("data", item)
            if isinstance(record, dict):
                records.append(record)

        total = payload.get("meta", {}).get("total", len(records))
        offset += len(page_data)
        logger.info("Fetched %d / %s records from %s", min(offset, total), total, endpoint)

        if offset >= total or len(page_data) < config.FDIC_API_PAGE_LIMIT:
            break

    return records


def download_failures(force: bool = False) -> pd.DataFrame:
    """Download FDIC failed-bank records and cache to data/raw/."""
    if config.RAW_FAILURES_FILE.exists() and not force:
        logger.info("Loading cached failures from %s", config.RAW_FAILURES_FILE.name)
        return pd.read_csv(config.RAW_FAILURES_FILE, parse_dates=["FAILDATE"])

    logger.info("Downloading FDIC failure records from API")
    params = {"fields": ",".join(config.FAILURE_FIELDS), "sort_by": "FAILDATE", "sort_order": "ASC"}
    records = _paginated_fetch(config.FDIC_FAILURES_ENDPOINT, params)
    if not records:
        raise DataLoaderError("FDIC failures API returned no records.")

    df = pd.DataFrame(records)
    df["FAILDATE"] = pd.to_datetime(df["FAILDATE"], errors="coerce")
    df = df.dropna(subset=["CERT", "FAILDATE"])
    df["CERT"] = df["CERT"].astype(int)

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_FAILURES_FILE, index=False)
    logger.info("Saved %d failure records to %s", len(df), config.RAW_FAILURES_FILE.name)
    return df


def download_financials(force: bool = False) -> pd.DataFrame:
    """Download quarterly FDIC financial snapshots and cache to data/raw/."""
    if config.RAW_FINANCIALS_FILE.exists() and not force:
        logger.info("Loading cached financials from %s", config.RAW_FINANCIALS_FILE.name)
        return pd.read_csv(config.RAW_FINANCIALS_FILE)

    logger.info("Downloading FDIC financial records for %d report dates", len(config.REPORT_DATES))
    all_records: list[dict[str, Any]] = []

    for repdte in config.REPORT_DATES:
        date_filter = f"REPDTE:{repdte}"
        params = {"filters": date_filter, "fields": ",".join(config.FINANCIAL_FIELDS), "sort_by": "CERT"}
        records = _paginated_fetch(config.FDIC_FINANCIALS_ENDPOINT, params)
        if not records:
            logger.warning("No financial records returned for REPDTE=%s", repdte)
            continue
        all_records.extend(records)

    if not all_records:
        raise DataLoaderError("FDIC financials API returned no records for configured report dates.")

    df = pd.DataFrame(all_records)
    df["REPDTE"] = df["REPDTE"].astype(str)
    df["CERT"] = pd.to_numeric(df["CERT"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["CERT"])
    df["CERT"] = df["CERT"].astype(int)

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_FINANCIALS_FILE, index=False)
    logger.info("Saved %d financial records to %s", len(df), config.RAW_FINANCIALS_FILE.name)
    return df


def _assign_failure_labels(financials: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    """Label each financial snapshot 1 if the bank fails within the lookahead window."""
    failure_map = failures.groupby("CERT")["FAILDATE"].min().to_dict()
    labeled = financials.copy()
    labeled["REPDTE_DT"] = pd.to_datetime(labeled["REPDTE"], format="%Y%m%d", errors="coerce")
    labeled = labeled.dropna(subset=["REPDTE_DT"])

    lookahead_days = config.FAILURE_LOOKAHEAD_QUARTERS * 91
    labels: list[int] = []

    for _, row in labeled.iterrows():
        cert = int(row["CERT"])
        repdte = row["REPDTE_DT"]
        fail_date = failure_map.get(cert)

        if fail_date is None or fail_date <= repdte:
            labels.append(0)
            continue

        days_to_failure = (fail_date - repdte).days
        labels.append(1 if 0 < days_to_failure <= lookahead_days else 0)

    labeled[config.TARGET_COLUMN] = labels
    labeled = labeled.drop(columns=["REPDTE_DT"])
    return labeled


def build_labeled_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Build a supervised dataset by merging FDIC financials with failure labels.

    Returns a DataFrame where each row is a (bank, quarter) observation with
    a binary ``failed`` target indicating failure within the lookahead window.
    """
    failures = download_failures(force=force_download)
    financials = download_financials(force=force_download)

    required_fin_cols = {"CERT", "REPDTE", "ASSET"}
    missing = required_fin_cols - set(financials.columns)
    if missing:
        raise DataLoaderError(f"Financial dataset missing required columns: {sorted(missing)}")

    if financials.empty:
        raise DataLoaderError("Financial dataset is empty.")

    dataset = _assign_failure_labels(financials, failures)
    positive_rate = dataset[config.TARGET_COLUMN].mean()
    logger.info(
        "Built labeled dataset: %d rows, %d banks, failure rate=%.4f",
        len(dataset),
        dataset["CERT"].nunique(),
        positive_rate,
    )

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(config.PROCESSED_DATASET_FILE, index=False)
    logger.info("Saved processed dataset to %s", config.PROCESSED_DATASET_FILE.name)
    return dataset


def load_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Load the bank failure dataset.

    Uses cached processed data when available; otherwise downloads from FDIC API.
    """
    if config.PROCESSED_DATASET_FILE.exists() and not force_download:
        logger.info("Loading processed dataset from %s", config.PROCESSED_DATASET_FILE.name)
        df = pd.read_csv(config.PROCESSED_DATASET_FILE)
        if df.empty:
            raise DataLoaderError("Processed dataset file is empty.")
        if config.TARGET_COLUMN not in df.columns:
            raise DataLoaderError(f"Processed dataset missing target column '{config.TARGET_COLUMN}'.")
        return df

    if config.RAW_FINANCIALS_FILE.exists() and config.RAW_FAILURES_FILE.exists() and not force_download:
        logger.info("Building labeled dataset from cached raw files")
        failures = pd.read_csv(config.RAW_FAILURES_FILE, parse_dates=["FAILDATE"])
        financials = pd.read_csv(config.RAW_FINANCIALS_FILE)
        dataset = _assign_failure_labels(financials, failures)
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(config.PROCESSED_DATASET_FILE, index=False)
        return dataset

    return build_labeled_dataset(force_download=force_download)


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate schema and basic quality of the loaded dataset."""
    if df is None or df.empty:
        raise DataLoaderError("Dataset is empty.")

    required = set(config.ID_COLUMNS + [config.TARGET_COLUMN, "ASSET"])
    missing = required - set(df.columns)
    if missing:
        raise DataLoaderError(f"Dataset missing required columns: {sorted(missing)}")

    if df[config.TARGET_COLUMN].nunique() < 2:
        raise DataLoaderError(
            "Dataset has only one class in the target column. "
            "Expand REPORT_DATES or FAILURE_LOOKAHEAD_QUARTERS in config."
        )

    if df["CERT"].isna().all():
        raise DataLoaderError("All CERT values are missing.")

    logger.info("Dataset validation passed: shape=%s", df.shape)


if __name__ == "__main__":
    from src.utils import ensure_directories, set_random_seeds

    ensure_directories()
    set_random_seeds()
    data = load_dataset(force_download=False)
    validate_dataset(data)
    print(f"Loaded dataset with shape {data.shape}, failure rate {data[config.TARGET_COLUMN].mean():.4f}")
