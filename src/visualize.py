import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    METRICS_DIR,
    FIGURES_DIR,
)

BASELINE_FILE = (
    METRICS_DIR /
    "baseline_metrics.csv"
)

DP_FILE = (
    METRICS_DIR /
    "dp_metrics.csv"
)

IMPORTANCE_FILE = (
    FIGURES_DIR.parent /
    "explainability" /
    "logistic_feature_importance.csv"
)


def plot_accuracy_vs_epsilon(
    dp_df,
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        dp_df["Epsilon"],
        dp_df["Accuracy"],
        marker="o",
    )

    plt.xlabel(
        "Epsilon"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Accuracy vs Epsilon"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "accuracy_vs_epsilon.png"
    )

    plt.close()


def plot_recall_vs_epsilon(
    dp_df,
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        dp_df["Epsilon"],
        dp_df["Recall"],
        marker="o",
    )

    plt.xlabel(
        "Epsilon"
    )

    plt.ylabel(
        "Recall"
    )

    plt.title(
        "Recall vs Epsilon"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "recall_vs_epsilon.png"
    )

    plt.close()


def plot_privacy_utility(
    dp_df,
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        dp_df["Epsilon"],
        dp_df["ROC_AUC"],
        marker="o",
    )

    plt.xlabel(
        "Epsilon"
    )

    plt.ylabel(
        "ROC AUC"
    )

    plt.title(
        "Privacy Utility Tradeoff"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "privacy_utility_tradeoff.png"
    )

    plt.close()


def plot_feature_importance():

    importance_df = pd.read_csv(
        IMPORTANCE_FILE
    )

    top_features = (
        importance_df
        .head(15)
        .sort_values(
            "absolute_coefficient"
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        top_features["feature"],
        top_features[
            "absolute_coefficient"
        ]
    )

    plt.xlabel(
        "Absolute Coefficient"
    )

    plt.title(
        "Top 15 Logistic Regression Features"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "logistic_feature_importance.png"
    )

    plt.close()


def plot_baseline_vs_dp(
    baseline_df,
    dp_df,
):

    baseline_auc = (
        baseline_df.loc[
            baseline_df["Model"]
            ==
            "Logistic Regression",
            "ROC_AUC",
        ]
        .iloc[0]
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        dp_df["Epsilon"],
        dp_df["ROC_AUC"],
        marker="o",
        label="DP Logistic",
    )

    plt.axhline(
        baseline_auc,
        linestyle="--",
        label="Baseline Logistic",
    )

    plt.xlabel(
        "Epsilon"
    )

    plt.ylabel(
        "ROC AUC"
    )

    plt.title(
        "Baseline vs DP Models"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "baseline_vs_dp_comparison.png"
    )

    plt.close()


def main():

    FIGURES_DIR.mkdir(
        exist_ok=True
    )

    baseline_df = pd.read_csv(
        BASELINE_FILE
    )

    dp_df = pd.read_csv(
        DP_FILE
    )

    plot_accuracy_vs_epsilon(
        dp_df
    )

    plot_recall_vs_epsilon(
        dp_df
    )

    plot_privacy_utility(
        dp_df
    )

    plot_feature_importance()

    plot_baseline_vs_dp(
        baseline_df,
        dp_df,
    )

    print(
        "Visualizations generated."
    )


if __name__ == "__main__":
    main()