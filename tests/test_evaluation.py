from pathlib import Path

from src.config import (
    METRICS_DIR,
)


def test_metrics_exist():

    baseline = (
        METRICS_DIR /
        "baseline_metrics.csv"
    )

    assert baseline.exists()