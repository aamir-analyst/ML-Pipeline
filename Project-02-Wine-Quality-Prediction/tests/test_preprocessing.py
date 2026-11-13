
from src.data.data_loader import load_data
from src.features.feature_engineering import create_features
from src.preprocessing.preprocess import (
    handle_missing_values,
    split_data
)


def test_missing_value_treatment():

    df = load_data()

    df = handle_missing_values(df)

    assert df.isnull().sum().sum() == 0


def test_train_test_split():

    df = load_data()

    X, y = create_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    assert len(X_train) == 1087
    assert len(X_test) == 272

    assert len(y_train) == 1087
    assert len(y_test) == 272

    assert X_train.shape[1] == 11
    assert X_test.shape[1] == 11