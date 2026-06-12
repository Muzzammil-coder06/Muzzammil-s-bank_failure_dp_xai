import pandas as pd

from src.data_loader import load_dataset
from src.feature_engineering import engineer_features
from src.preprocess import preprocess_pipeline


def test_preprocess_pipeline():

    df = load_dataset()

    featured, _ = engineer_features(df)

    result = preprocess_pipeline(
        featured,
        save=False,
    )

    assert result.X_train.shape[0] > 0