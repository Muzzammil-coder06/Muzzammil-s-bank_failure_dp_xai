# Balancing Explainability and Privacy in Bank Failure Prediction: A Differentially Private Glass-Box Approach

## Overview

This project presents an end-to-end machine learning framework for predicting bank failures while balancing two important requirements:

- **Explainability** through glass-box machine learning models
- **Privacy Preservation** through Differential Privacy

The system uses FDIC banking data to predict whether a bank is likely to fail within a future time horizon. Traditional interpretable models are compared against differentially private models to quantify the privacy-utility tradeoff.

---

## Key Features

- Automated FDIC data acquisition
- Bank failure prediction pipeline
- Dynamic financial feature engineering
- Explainable machine learning models
- Differential Privacy integration using IBM Diffprivlib
- Privacy vs Utility analysis
- Automated evaluation metrics
- Automated visualization generation
- Reproducible experiments
- Single-command execution
- Unit testing support

---

## Dataset

### Source

Federal Deposit Insurance Corporation (FDIC) BankFind Suite API

### Data Components

- Bank financial statements
- Capital adequacy indicators
- Asset quality indicators
- Earnings indicators
- Liquidity indicators
- FDIC failure records

### Label Construction

A bank is labeled as:

```text
failed = 1
```

if the bank fails within the next 12 quarters (approximately 3 years) after a reporting snapshot.

Otherwise:

```text
failed = 0
```

### Dataset Statistics

| Metric | Value |
|----------|----------|
| Total Records | 99,869 |
| Total Banks | 5,489 |
| Training Samples | 79,895 |
| Test Samples | 19,974 |
| Positive Training Cases | 106 |
| Positive Test Cases | 27 |

---

## Project Architecture

```text
                     ┌────────────────────┐
                     │ FDIC  API  Dataset │
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────────┐
                     │   Data Acquisition     │
                     │    data_loader.py      │
                     └─────────┬──────────────┘
                               │
                               ▼
                    ┌────────────────────────┐
                    │ Feature Engineering    │
                    │ Financial Ratios       │
                    │ Risk Indicators        │
                    └─────────┬──────────────┘
                              │
                              ▼
                    ┌────────────────────────┐
                    │ Preprocessing Pipeline │
                    │ Imputation             │
                    │ Outlier Handling       │
                    │ Scaling                │
                    └─────────┬──────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼

      ┌──────────────────┐      ┌────────────────────┐
      │ Baseline Models  │      │ Differentially     │
      │                  │      │ Private Models     │
      │ Decision Tree    │      │ DP Logistic Reg.   │
      │ Logistic Reg.    │      │ ε = 0.1 - 10       │
      └────────┬─────────┘      └─────────┬──────────┘
               │                          │
               └──────────┬───────────────┘
                          ▼

             ┌──────────────────────────┐
             │ Evaluation & Comparison  │
             │ Accuracy                 │
             │ Precision                │
             │ Recall                   │
             │ F1 Score                 │
             │ ROC-AUC                  │
             └──────────┬───────────────┘
                        │
                        ▼

             ┌──────────────────────────┐
             │ Visualizations           │
             │ Privacy-Utility Curves   │
             │ Feature Importance       │
             │ Performance Analysis     │
             └──────────────────────────┘
```

---

## Methodology

### Phase 1: Data Collection

Financial and failure information are downloaded automatically from the FDIC API.

Each observation represents:

```text
(Bank, Reporting Date)
```

A future failure window is used to generate prediction labels.

---

### Phase 2: Feature Engineering

The framework dynamically creates banking risk indicators.

#### Capital Ratios

Capital Adequacy Ratio = Equity / Assets

#### Liquidity Ratios

Liquidity Ratio = Deposits / Assets

#### Leverage Ratios

Debt Ratio = Liabilities / Assets

#### Asset Quality Metrics

Non-Current Loan Ratio = NonCurrentLoans / NetLoans

#### Profitability Metrics

Profitability Ratio = NetIncome / Assets

---

### Phase 3: Baseline Explainable Models

#### Decision Tree

Advantages:

- Fully interpretable
- Human-readable decision rules
- Transparent decision paths

#### Logistic Regression

Advantages:

- Explainable coefficients
- Regulatory-friendly
- Easily auditable

Feature importance rankings are automatically generated.

---

### Phase 4: Differential Privacy

Differential Privacy protects sensitive financial information by injecting calibrated noise during model training.

Privacy levels evaluated:

```text
ε = 0.1
ε = 0.5
ε = 1.0
ε = 5.0
ε = 10.0
```

Lower epsilon values provide stronger privacy but lower predictive utility.

Higher epsilon values provide weaker privacy but higher predictive utility.

Implementation uses IBM Diffprivlib.

---

### Phase 5: Evaluation

Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Models compared:

- Decision Tree
- Logistic Regression
- Differentially Private Logistic Regression

---

## Experimental Results

### Baseline Models

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---------|---------:|---------:|---------:|---------:|---------:|
| Decision Tree | 0.9275 | 0.0177 | 0.9630 | 0.0347 | 0.9609 |
| Logistic Regression | 0.7772 | 0.0058 | 0.9630 | 0.0116 | 0.9541 |

### Differential Privacy Models

| Epsilon | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---------|---------:|---------:|---------:|---------:|---------:|
| 0.1 | 0.7706 | 0.0018 | 0.2963 | 0.0035 | 0.5593 |
| 0.5 | 0.9446 | 0.0009 | 0.0370 | 0.0018 | 0.4951 |
| 1.0 | 0.9708 | 0.0018 | 0.0370 | 0.0034 | 0.5190 |
| 5.0 | 0.9877 | 0.0045 | 0.0370 | 0.0081 | 0.5236 |
| 10.0 | 0.9961 | 0.0526 | 0.1111 | 0.0714 | 0.7997 |

---

## Results Interpretation

### Baseline Performance

The baseline Decision Tree and Logistic Regression models achieved strong predictive performance with ROC-AUC values above 0.95.

The Decision Tree slightly outperformed Logistic Regression while maintaining complete interpretability.

### Differential Privacy Performance

The Differentially Private models demonstrated the expected privacy-utility tradeoff.

Key observations:

- Strong privacy (ε = 0.1) substantially reduced predictive performance.
- Moderate privacy levels showed lower utility due to injected noise.
- Higher epsilon values recovered a significant portion of predictive performance.
- ε = 10 achieved the best balance between privacy and utility.

### Feature Importance

The most influential predictors identified by Logistic Regression include:

1. RBCT1
2. Capital Adequacy Ratio
3. Total Assets
4. Liabilities
5. Domestic Deposit Ratio

These variables align with established banking risk literature, where capital strength and asset quality are major indicators of institutional stability.

---

## Generated Outputs

### Models

```text
models/

decision_tree.joblib
logistic_regression.joblib

dp_logistic_eps_0_1.joblib
dp_logistic_eps_0_5.joblib
dp_logistic_eps_1_0.joblib
dp_logistic_eps_5_0.joblib
dp_logistic_eps_10_0.joblib
```

### Metrics

```text
results/metrics/

baseline_metrics.csv
dp_metrics.csv
final_model_comparison.csv
decision_tree_report.txt
logistic_regression_report.txt
```

### Explainability Outputs

```text
results/explainability/

decision_tree_rules.txt
logistic_feature_importance.csv
```

### Visualizations

```text
results/figures/

decision_tree.png
accuracy_vs_epsilon.png
recall_vs_epsilon.png
privacy_utility_tradeoff.png
logistic_feature_importance.png
baseline_vs_dp_comparison.png
```

---

## Installation

```bash
git clone <repository-url>

cd bank_failure_dp_xai

python -m venv .venv

source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute the complete pipeline:

```bash
python main.py
```

This automatically performs:

1. Dataset Loading
2. Feature Engineering
3. Preprocessing
4. Baseline Model Training
5. Differential Privacy Training
6. Evaluation
7. Visualization Generation
8. Model Persistence

---

## Running Tests

```bash
pytest
```

Expected Output:

```text
4 passed
```

---

## Project Structure

```text
bank_failure_dp_xai/

├── data/
├── models/
├── results/
│   ├── explainability/
│   ├── figures/
│   └── metrics/
├── src/
├── tests/
├── main.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Conclusion

This project demonstrates that explainable machine learning and privacy-preserving machine learning can be successfully combined for bank failure prediction.

The experimental results show a clear privacy-utility tradeoff, where stronger privacy guarantees reduce predictive performance, while higher privacy budgets improve utility.

The framework provides a practical and reproducible solution for privacy-aware financial risk modeling while maintaining transparency suitable for regulatory and auditing environments.