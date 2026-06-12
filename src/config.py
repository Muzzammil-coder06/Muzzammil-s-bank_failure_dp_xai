"""Central configuration for the bank failure prediction pipeline."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# FDIC API
# ---------------------------------------------------------------------------
FDIC_API_BASE = "https://api.fdic.gov/banks"
FDIC_FINANCIALS_ENDPOINT = f"{FDIC_API_BASE}/financials"
FDIC_FAILURES_ENDPOINT = f"{FDIC_API_BASE}/failures"
FDIC_API_PAGE_LIMIT = 10_000
FDIC_API_TIMEOUT_SECONDS = 60
FDIC_API_RETRY_ATTEMPTS = 3
FDIC_API_RETRY_DELAY_SECONDS = 2.0

# Raw file names (cached after first download)
RAW_FAILURES_FILE = RAW_DATA_DIR / "fdic_failures.csv"
RAW_FINANCIALS_FILE = RAW_DATA_DIR / "fdic_financials.csv"
PROCESSED_DATASET_FILE = PROCESSED_DATA_DIR / "bank_failure_dataset.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"
SPLIT_METADATA_FILE = PROCESSED_DATA_DIR / "split_metadata.csv"
IMPUTER_FILE = PROCESSED_DATA_DIR / "imputer.joblib"
SCALER_FILE = PROCESSED_DATA_DIR / "scaler.joblib"
FEATURE_COLUMNS_FILE = PROCESSED_DATA_DIR / "feature_columns.joblib"

# Quarterly report dates used for financial snapshots (YYYY-MM-DD → YYYYMMDD)
REPORT_DATES: list[str] = [
    "20150331",
    "20150630",
    "20150930",
    "20151231",
    "20160331",
    "20160630",
    "20160930",
    "20161231",
    "20170331",
    "20170630",
    "20170930",
    "20171231",
    "20180331",
    "20180630",
    "20180930",
    "20181231",
    "20190331",
    "20190630",
    "20190930",
    "20191231",
    "20200331",
    "20200630",
    "20200930",
    "20201231",
    "20210331",
    "20210630",
    "20210930",
    "20211231",
    "20220331",
    "20220630",
    "20220930",
    "20221231",
    "20230331",
    "20230630",
    "20230930",
    "20231231",
]

# Financial fields requested from the FDIC API (CAMELS-aligned ratios)
FINANCIAL_FIELDS: list[str] = [
    "CERT",
    "REPDTE",
    "ASSET",
    "DEP",
    "DEPDOM",
    "EQ",
    "NETINC",
    "INTINC",
    "LIAB",
    "ROA",
    "ROE",
    "RBCT1",
    "RBC1AAJ",
    "RBC1RWAJ",
    "ELNANTR",
    "ORE",
    "LNLSNET",
    "LNLSGR",
    "NTLNLS",
    "LNATRES",
    "NCLNLS",
    "LNRESNCR",
    "EEFFR",
    "ERNASTR",
]

FAILURE_FIELDS: list[str] = [
    "CERT",
    "FAILDATE",
    "NAME",
    "CITYST",
    "QBFASSET",
    "QBFDEP",
]

# Quarters ahead to label a bank as "failed" after a financial snapshot
FAILURE_LOOKAHEAD_QUARTERS = 12

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
OUTLIER_IQR_MULTIPLIER = 3.0
MIN_SAMPLES_PER_CLASS = 5

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
DECISION_TREE_PARAMS: dict = {
    "max_depth": 6,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

LOGISTIC_REGRESSION_PARAMS: dict = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "solver": "lbfgs",
}

# Differential privacy epsilon values
EPSILON_VALUES: list[float] = [0.1, 0.5, 1.0, 5.0, 10.0]
DP_METRICS_FILE = METRICS_DIR / "dp_metrics.csv"
DP_DATA_NORM = 35.0

DP_LOGISTIC_PARAMS: dict = {
    "max_iter": 1000,
    "class_weight": "balanced",
}

DP_DECISION_TREE_PARAMS: dict = {
    "max_depth": 6,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
}

# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------
TARGET_COLUMN = "failed"
ID_COLUMNS: list[str] = ["CERT", "REPDTE"]

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

CLASSIFICATION_THRESHOLD = 0.30

DECISION_TREE_PARAMS: dict = {
    "max_depth": 5,
    "min_samples_leaf": 50,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

LOGISTIC_REGRESSION_PARAMS: dict = {
    "max_iter": 5000,
    "solver": "liblinear",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

# Differential privacy epsilon values
EPSILON_VALUES: list[float] = [
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
]

DP_LOGISTIC_PARAMS: dict = {
    "max_iter": 1000,
    "class_weight": "balanced",
}

DP_DECISION_TREE_PARAMS: dict = {
    "max_depth": 5,
    "min_samples_leaf": 50,
    "class_weight": "balanced",
}

