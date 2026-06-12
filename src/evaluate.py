import pandas as pd

from src.config import (
    METRICS_DIR,
)

BASELINE_FILE = (
    METRICS_DIR /
    "baseline_metrics.csv"
)

DP_FILE = (
    METRICS_DIR /
    "dp_metrics.csv"
)

OUTPUT_FILE = (
    METRICS_DIR /
    "final_model_comparison.csv"
)


def main():

    baseline_df = pd.read_csv(
        BASELINE_FILE
    )

    dp_df = pd.read_csv(
        DP_FILE
    )

    if "Epsilon" not in baseline_df.columns:
        baseline_df["Epsilon"] = "N/A"

    final_df = pd.concat(
        [
            baseline_df,
            dp_df,
        ],
        ignore_index=True,
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(final_df)


if __name__ == "__main__":
    main()