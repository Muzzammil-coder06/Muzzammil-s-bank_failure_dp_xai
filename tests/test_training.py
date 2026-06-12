import pandas as pd

from src.train_baseline import (
    train_decision_tree,
)


def test_decision_tree_training():

    X = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [2, 3, 4, 5],
        }
    )

    y = [0, 0, 1, 1]

    model = train_decision_tree(
        X,
        y,
    )

    assert model is not None