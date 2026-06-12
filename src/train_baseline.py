"""
Baseline model training for bank failure prediction.

Models:
1. Decision Tree
2. Logistic Regression

Outputs:
- Saved models
- Metrics
- Classification reports
- Decision tree rules
- Feature importance rankings
- Tree visualization
"""

from pathlib import Path
import logging

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import (
    DecisionTreeClassifier,
    export_text,
    plot_tree,
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from src.config import (
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    FEATURE_COLUMNS_FILE,
    TARGET_COLUMN,
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    DECISION_TREE_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    CLASSIFICATION_THRESHOLD,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

LOGGER = logging.getLogger(__name__)

EXPLAINABILITY_DIR = RESULTS_DIR / "explainability"


def create_directories() -> None:
    """Create required output directories."""

    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    EXPLAINABILITY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_data():
    """Load processed train/test datasets."""

    train_df = pd.read_csv(
        TRAIN_DATA_FILE
    )

    test_df = pd.read_csv(
        TEST_DATA_FILE
    )

    feature_columns = joblib.load(
        FEATURE_COLUMNS_FILE
    )

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        TARGET_COLUMN
    ]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns,
    )


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name,
):
    """
    Evaluate a model.
    """

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    y_pred = (
        y_prob >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    metrics = {
        "Model": model_name,
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

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    return (
        metrics,
        report,
    )


def save_model(
    model,
    filename,
):
    """Persist model."""

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


def save_metrics(
    metrics_list,
):
    """Save baseline metrics table."""

    metrics_df = pd.DataFrame(
        metrics_list
    )

    output_path = (
        METRICS_DIR /
        "baseline_metrics.csv"
    )

    metrics_df.to_csv(
        output_path,
        index=False,
    )

    LOGGER.info(
        "Saved metrics: %s",
        output_path,
    )


def save_classification_report(
    report_text,
    filename,
):
    """Save classification report."""

    output_path = (
        METRICS_DIR /
        filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            report_text
        )


def save_tree_rules(
    model,
    feature_names,
):
    """Export decision tree rules."""

    rules = export_text(
        model,
        feature_names=list(
            feature_names
        ),
    )

    output_path = (
        EXPLAINABILITY_DIR /
        "decision_tree_rules.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            rules
        )

    LOGGER.info(
        "Saved tree rules."
    )


def save_tree_visualization(
    model,
    feature_names,
):
    """Create decision tree plot."""

    plt.figure(
        figsize=(22, 12)
    )

    plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        fontsize=6,
        max_depth=4,
    )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR /
        "decision_tree.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    LOGGER.info(
        "Saved tree visualization."
    )


def save_logistic_importance(
    model,
    feature_names,
):
    """Save logistic regression coefficients."""

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_[0],
        }
    )

    coef_df[
        "absolute_coefficient"
    ] = coef_df[
        "coefficient"
    ].abs()

    coef_df = coef_df.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    output_path = (
        EXPLAINABILITY_DIR /
        "logistic_feature_importance.csv"
    )

    coef_df.to_csv(
        output_path,
        index=False,
    )

    LOGGER.info(
        "Saved coefficient ranking."
    )


def train_decision_tree(
    X_train,
    y_train,
):
    """Train Decision Tree."""

    model = DecisionTreeClassifier(
        **DECISION_TREE_PARAMS
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def train_logistic_regression(
    X_train,
    y_train,
):
    """Train Logistic Regression."""

    model = LogisticRegression(
        **LOGISTIC_REGRESSION_PARAMS
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def main():

    create_directories()

    LOGGER.info(
        "Loading processed data..."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    ) = load_data()

    LOGGER.info(
        "Training Decision Tree..."
    )

    decision_tree = train_decision_tree(
        X_train,
        y_train,
    )

    LOGGER.info(
        "Training Logistic Regression..."
    )

    logistic_model = (
        train_logistic_regression(
            X_train,
            y_train,
        )
    )

    save_model(
        decision_tree,
        "decision_tree.joblib",
    )

    save_model(
        logistic_model,
        "logistic_regression.joblib",
    )

    dt_metrics, dt_report = (
        evaluate_model(
            decision_tree,
            X_test,
            y_test,
            "Decision Tree",
        )
    )

    lr_metrics, lr_report = (
        evaluate_model(
            logistic_model,
            X_test,
            y_test,
            "Logistic Regression",
        )
    )

    save_metrics(
        [
            dt_metrics,
            lr_metrics,
        ]
    )

    save_classification_report(
        dt_report,
        "decision_tree_report.txt",
    )

    save_classification_report(
        lr_report,
        "logistic_regression_report.txt",
    )

    save_tree_rules(
        decision_tree,
        feature_names,
    )

    save_tree_visualization(
        decision_tree,
        feature_names,
    )

    save_logistic_importance(
        logistic_model,
        feature_names,
    )

    metrics_df = pd.DataFrame(
        [
            dt_metrics,
            lr_metrics,
        ]
    )

    print("\n")
    print(metrics_df)
    print("\n")

    LOGGER.info(
        "Phase 3 completed successfully."
    )


if __name__ == "__main__":
    main()