"""Validate the customer data before preprocessing."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.common import RAW_COLUMNS, ROOT, TARGET


def validate_file(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = pd.read_csv(path)
    missing_columns = sorted(set(RAW_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")
    if data.empty:
        raise ValueError("Dataset is empty")
    if data[TARGET].isna().any() or not data[TARGET].isin([0, 1]).all():
        raise ValueError("ProdTaken must contain 0 or 1 without missing values")
    if data["CustomerID"].duplicated().any():
        raise ValueError("Customer IDs must be unique")

    outcome_counts = data[TARGET].value_counts().reindex([0, 1], fill_value=0)
    if outcome_counts.min() < 4:
        raise ValueError("Both target classes need at least four examples")

    try:
        dataset_location = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        dataset_location = path.name

    summary = {
        "dataset": dataset_location,
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "missing_cells": int(data.isna().sum().sum()),
        "duplicate_customer_ids": int(data["CustomerID"].duplicated().sum()),
        "target_distribution": {
            str(label): int(count) for label, count in outcome_counts.items()
        },
        "purchase_rate": round(float(data[TARGET].mean()), 6),
        "unrecognized_columns": sorted(
            set(data.columns) - set(RAW_COLUMNS) - {"Unnamed: 0"}
        ),
        "validation_status": "PASS",
    }
    print("Data validation summary")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "tourism.csv")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports" / "data_validation.json"
    )
    args = parser.parse_args()

    summary = validate_file(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
