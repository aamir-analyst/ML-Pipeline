
import pandas as pd

from src.pipeline.predict import (
    load_model,
    predict_wine,
    FEATURES
)


def test_prediction():

    model = load_model()

    # Sample wine data
    wine_data = pd.DataFrame(
        [[
            7.4,
            0.50,
            0.30,
            2.0,
            0.08,
            15,
            40,
            0.996,
            3.30,
            0.70,
            12.0
        ]],
        columns=FEATURES
    )

    result = predict_wine(model, wine_data)

    assert result in ["Premium Wine", "Non-Premium Wine"]
