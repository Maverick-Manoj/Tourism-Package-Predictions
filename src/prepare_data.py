"""Prepare customer data for training and evaluation."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import FEATURE_COLUMNS, ROOT, TARGET, clean_data
from src.validate_data import validate_file


def prepare(
    data_path: Path,
    output_dir: Path,
    test_size: float = 0.20,
    seed: int = 42,
):
    validate_file(data_path)
    cleaned = clean_data(pd.read_csv(data_path))
    train, test = train_test_split(
        cleaned,
        test_size=test_size,
        random_state=seed,
        stratify=cleaned[TARGET],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    try:
        output_location = output_dir.resolve().relative_to(ROOT.resolve())
    except ValueError:
        output_location = Path(output_dir.name)

    summary = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_positive_rate": round(float(train[TARGET].mean()), 6),
        "test_positive_rate": round(float(test[TARGET].mean()), 6),
        "feature_columns": FEATURE_COLUMNS,
        "train_file": (output_location / "train.csv").as_posix(),
        "test_file": (output_location / "test.csv").as_posix(),
        "test_size": test_size,
        "random_state": seed,
    }
    report_file = output_dir / "split_manifest.json"
    report_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Data preparation summary")
    print(json.dumps(summary, indent=2))
    return train, test, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "tourism.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    prepare(args.data, args.output_dir)


if __name__ == "__main__":
    main()