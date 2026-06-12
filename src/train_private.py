"""
Differentially Private Bank Failure Prediction

Trains DP Logistic Regression models
for multiple epsilon values and saves:

- Models
- Metrics
- Comparison table
"""

import logging

import joblib
import pandas as pd

from diffprivlib.models import LogisticRegression as DPLogisticRegression  # type: ignore

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.config import (
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    FEATURE_COLUMNS_FILE,
    TARGET_COLUMN,
    MODELS_DIR,
    METRICS_DIR,
    DP_METRICS_FILE,
    EPSILON_VALUES,
    DP_DATA_NORM,
    RANDOM_SEED,
    CLASSIFICATION_THRESHOLD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

LOGGER = logging.getLogger(__name__)


def load_data():
    """Load processed datasets."""

    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)

    feature_columns = joblib.load(
        FEATURE_COLUMNS_FILE
    )

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def evaluate_model(
    model,
    X_test,
    y_test,
):
    """Evaluate DP model."""

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    y_pred = (
        y_prob >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    return {
        "Accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "F1": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "ROC_AUC": roc_auc_score(
            y_test,
            y_prob,
        ),
    }


def train_dp_model(
    epsilon,
    X_train,
    y_train,
):
    """Train a DP Logistic Regression model."""

    model = DPLogisticRegression(
        epsilon=epsilon,
        data_norm=DP_DATA_NORM,
        max_iter=1000,
        random_state=RANDOM_SEED,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def save_model(
    model,
    epsilon,
):
    """Persist model."""

    filename = (
        f"dp_logistic_eps_"
        f"{str(epsilon).replace('.', '_')}.joblib"
    )

    output_path = (
        MODELS_DIR /
        filename
    )

    joblib.dump(
        model,
        output_path,
    )

    LOGGER.info(
        "Saved model: %s",
        output_path,
    )


def main():

    MODELS_DIR.mkdir(
        exist_ok=True
    )

    METRICS_DIR.mkdir(
        exist_ok=True
    )

    LOGGER.info(
        "Loading processed data..."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_data()

    results = []

    for epsilon in EPSILON_VALUES:

        LOGGER.info(
            "Training epsilon=%s",
            epsilon,
        )

        model = train_dp_model(
            epsilon,
            X_train,
            y_train,
        )

        save_model(
            model,
            epsilon,
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        metrics["Model"] = (
            "DP Logistic Regression"
        )

        metrics["Epsilon"] = epsilon

        results.append(
            metrics
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df[
        [
            "Model",
            "Epsilon",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
        ]
    ]

    results_df.to_csv(
        DP_METRICS_FILE,
        index=False,
    )

    print("\n")
    print(results_df)
    print("\n")

    LOGGER.info(
        "Saved DP metrics: %s",
        DP_METRICS_FILE,
    )

    LOGGER.info(
        "Phase 4 completed successfully."
    )


if __name__ == "__main__":
    main()