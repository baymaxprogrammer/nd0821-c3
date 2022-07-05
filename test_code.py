"""
Unit test for ML model
"""
import pandas as pd
import pytest
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from training.ml.data import process_data

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# Using fixture helps reusing data as a variable (here pandas dataframe)
@pytest.fixture()
def data():
    # Check if the path and file exists
    assert os.path.exists('./data/census.csv')
    return pd.read_csv('./data/census.csv')


def test_data_and_features(data):
    # Data should have at least 100 samples
    assert data.shape[0] > 100

    # Data should contain 15 features
    assert data.shape[1] == 15

    # Check if all the categorical features represented in the data
    for cat_feat in cat_features:
        assert cat_feat in list(data.columns)


def test_process_data_function(data):
    train, test = train_test_split(data, random_state=42, test_size=0.2)
    X, y, _, _ = process_data(
        train, cat_features, label='salary'
    )
    assert len(X) == len(y)


def test_model_and_paths():
    # Check if all the model files exist
    assert os.path.exists("./model/model.pkl")
    assert os.path.exists("./model/encoder.pkl")
    assert os.path.exists("./model/lb.pkl")

    # Load the model
    model = joblib.load("./model/model.pkl")
    assert isinstance(model, RandomForestClassifier)
