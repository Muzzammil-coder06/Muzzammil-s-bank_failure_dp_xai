"""
Main orchestration pipeline.

Runs the entire bank failure prediction workflow:

1. Load dataset
2. Feature engineering
3. Preprocessing
4. Train baseline models
5. Train DP models
6. Evaluate results
7. Generate visualizations
"""

import logging

from src.data_loader import load_dataset
from src.feature_engineering import engineer_features
from src.preprocess import preprocess_pipeline

from src.train_baseline import (
    main as train_baseline_main,
)

from src.train_private import (
    main as train_private_main,
)

from src.evaluate import (
    main as evaluate_main,
)

from src.visualize import (
    main as visualize_main,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

LOGGER = logging.getLogger(__name__)


def main():

    LOGGER.info(
        "Step 1: Loading dataset..."
    )

    dataset = load_dataset()

    LOGGER.info(
        "Step 2: Feature engineering..."
    )

    featured_df, _ = engineer_features(
        dataset
    )

    LOGGER.info(
        "Step 3: Preprocessing..."
    )

    preprocess_pipeline(
        featured_df,
        save=True,
    )

    LOGGER.info(
        "Step 4: Training baseline models..."
    )

    train_baseline_main()

    LOGGER.info(
        "Step 5: Training DP models..."
    )

    train_private_main()

    LOGGER.info(
        "Step 6: Evaluation..."
    )

    evaluate_main()

    LOGGER.info(
        "Step 7: Visualization..."
    )

    visualize_main()

    LOGGER.info(
        "Pipeline completed successfully."
    )


if __name__ == "__main__":
    main()