import os
import pytest

@pytest.mark.parametrize("filename", [
    "logistic_regression_model.pkl",
    "random_forest_model.pkl"
])
def test_model_file_exists(filename):
    assert os.path.exists(filename), f"{filename} does not exist!"
