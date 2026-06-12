"""Shared utilities: logging, seeding, and path helpers."""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT, RANDOM_SEED


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project root logger."""
    logger = logging.getLogger("bank_failure_dp_xai")
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories() -> None:
    """Create all required project directories if they do not exist."""
    from src.config import (
        FIGURES_DIR,
        METRICS_DIR,
        MODELS_DIR,
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
        RESULTS_DIR,
    )

    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_project_path(relative: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    return (PROJECT_ROOT / relative).resolve()
