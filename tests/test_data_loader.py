from src.data_loader import load_dataset


def test_dataset_loads():

    df = load_dataset()

    assert len(df) > 0