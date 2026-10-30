"""Basic tests for data preparation and model predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.common import FEATURE_COLUMNS, RAW_COLUMNS, clean_data
from src.train_model import make_pipeline
from src.validate_data import validate_file


def sample_raw():
    return pd.read_csv("data/tourism.csv", nrows=40)


def test_required_schema():
    assert len(RAW_COLUMNS) == 20
    assert len(FEATURE_COLUMNS) == 13


def test_gender_and_marital_status_cleaned():
    data = sample_raw()
    data.loc[0, "Gender"] = "Fe Male"
    data.loc[0, "MaritalStatus"] = "Unmarried"

    cleaned = clean_data(data)
    assert cleaned.loc[0, "Gender"] == "Female"
    assert cleaned.loc[0, "MaritalStatus"] == "Single"


def test_contact_details_not_used_for_prediction():
    cleaned = clean_data(sample_raw())
    contact_columns = [
        "CustomerID", "DurationOfPitch", "PitchSatisfactionScore",
        "NumberOfFollowups", "ProductPitched", "TypeofContact", "Unnamed: 0",
    ]
    for column in contact_columns:
        assert column not in cleaned.columns


def test_missing_target_is_rejected():
    with pytest.raises(ValueError, match="ProdTaken"):
        clean_data(sample_raw().drop(columns=["ProdTaken"]))


def test_pipeline_predicts_probabilities():
    data = clean_data(sample_raw())
    model = make_pipeline(DecisionTreeClassifier(max_depth=3, random_state=42))
    model.fit(data[FEATURE_COLUMNS], data["ProdTaken"])

    scores = model.predict_proba(data[FEATURE_COLUMNS])[:, 1]
    assert len(scores) == len(data)
    assert np.all((scores >= 0) & (scores <= 1))


def test_dataset_passes_validation():
    summary = validate_file(Path("data/tourism.csv"))
    assert summary["validation_status"] == "PASS"