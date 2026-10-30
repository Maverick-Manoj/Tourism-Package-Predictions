"""Evaluate a saved model against the held-out test records."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.common import FEATURE_COLUMNS, ROOT, TARGET


def evaluate(model_path: Path, test_path: Path):
    model = joblib.load(model_path)
    test = pd.read_csv(test_path)
    scores = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    if not ((scores >= 0).all() and (scores <= 1).all()):
        raise ValueError("Predicted probabilities are outside the [0, 1] range")

    metrics = {
        "holdout_pr_auc": float(average_precision_score(test[TARGET], scores)),
        "holdout_roc_auc": float(roc_auc_score(test[TARGET], scores)),
        "num_records": len(test),
        "num_scores": len(scores),
    }
    print("Saved model evaluation")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models" / "best_model.joblib"
    )
    parser.add_argument("--test", type=Path, default=ROOT / "build" / "test.csv")
    args = parser.parse_args()
    evaluate(args.model, args.test)


if __name__ == "__main__":
    main()