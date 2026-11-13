
from src.data.data_loader import load_data
from src.features.feature_engineering import create_features


def test_create_features():

    df = load_data()

    X, y = create_features(df)

    # Features should contain 11 columns
    assert X.shape[1] == 11

    # Target should contain 1359 rows
    assert len(y) == 1359

    # Target should contain only 0 and 1
    assert set(y.unique()).issubset({0, 1})

    # Original target should not be in features
    assert "quality" not in X.columns

    # Premium target should not be in features
    assert "is_premium" not in X.columns
