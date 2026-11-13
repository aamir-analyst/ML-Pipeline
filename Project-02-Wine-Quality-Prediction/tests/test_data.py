
from src.data.data_loader import load_data


def test_load_data():

    df = load_data()

    # Dataset should not be empty
    assert not df.empty

    # Dataset should contain 12 columns
    assert df.shape[1] == 12

    # Required target column
    assert "quality" in df.columns

    # Dataset should not contain duplicate rows
    assert df.duplicated().sum() == 0