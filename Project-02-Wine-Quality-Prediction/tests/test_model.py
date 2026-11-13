
from src.data.data_loader import load_data
from src.features.feature_engineering import create_features
from src.preprocessing.preprocess import split_data
from src.models.train import train_models


def test_model_training():

    df = load_data()

    X, y = create_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    models = train_models(X_train, y_train)

    # Check all five models
    assert len(models) == 5

    # Check model names
    expected_models = {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "SVM"
    }

    assert set(models.keys()) == expected_models

    # Check every model can predict
    for model in models.values():

        predictions = model.predict(X_test)

        assert len(predictions) == len(y_test)

        assert set(predictions).issubset({0, 1})
