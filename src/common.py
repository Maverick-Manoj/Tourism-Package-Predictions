"""Column definitions and shared data cleaning functions."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGET = "ProdTaken"

RAW_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

DROP_COLUMNS = [
    "Unnamed: 0", "CustomerID", "TypeofContact", "DurationOfPitch",
    "NumberOfFollowups", "ProductPitched", "PitchSatisfactionScore",
]

FEATURE_COLUMNS = [
    column for column in RAW_COLUMNS
    if column not in DROP_COLUMNS and column != TARGET
]
CATEGORICAL_COLUMNS = ["Occupation", "Gender", "MaritalStatus", "Designation"]
NUMERIC_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS
]


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean customer categories and select the inputs available for prediction."""
    missing = sorted(set(RAW_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    cleaned = data.drop_duplicates(subset="CustomerID", keep="first").copy()
    cleaned["Gender"] = cleaned["Gender"].replace({"Fe Male": "Female"})
    cleaned["MaritalStatus"] = cleaned["MaritalStatus"].replace(
        {"Unmarried": "Single"}
    )

    for column in NUMERIC_COLUMNS + [TARGET]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if cleaned[TARGET].isna().any() or not cleaned[TARGET].isin([0, 1]).all():
        raise ValueError("ProdTaken must contain only 0 and 1")

    return cleaned[FEATURE_COLUMNS + [TARGET]]