"""Compare classifiers and save the best cross-validated model."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.common import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    ROOT,
    TARGET,
)


def json_value(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Value cannot be converted to JSON: {type(value)}")


def make_pipeline(classifier):
    numeric_processing = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])
    category_processing = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_processing, NUMERIC_COLUMNS),
        ("category", category_processing, CATEGORICAL_COLUMNS),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def candidate_models():
    random_seed = 42

    return {
        "DecisionTree": (
            DecisionTreeClassifier(random_state=random_seed),
            {
                "classifier__max_depth": [4, 8],
                "classifier__min_samples_leaf": [2, 8],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=random_seed, n_jobs=1),
            {
                "classifier__n_estimators": [100, 180],
                "classifier__max_depth": [6, None],
                "classifier__min_samples_leaf": [2],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=random_seed),
            {
                "classifier__n_estimators": [100, 150],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [2],
            },
        ),
    }


def experiment(train_path, test_path, model_dir, report_dir, tracking_dir):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    expected_columns = set(FEATURE_COLUMNS + [TARGET])
    for split_name, split_data in [("training", train), ("test", test)]:
        if set(split_data.columns) != expected_columns:
            raise ValueError(
                f"Unexpected or missing columns in {split_name} data"
            )

    training_features = train[FEATURE_COLUMNS]
    training_target = train[TARGET].astype(int)
    test_features = test[FEATURE_COLUMNS]
    test_target = test[TARGET].astype(int)

    for folder in [model_dir, report_dir, tracking_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    try:
        import mlflow
    except ImportError:
        mlflow = None

    if mlflow is not None:
        mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
        mlflow.set_experiment("tourism-purchase-prediction")

    cross_validation = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=42
    )
    trials = []
    comparisons = []
    best_result = None

    for name, (classifier, parameter_grid) in candidate_models().items():
        search = GridSearchCV(
            estimator=make_pipeline(classifier),
            param_grid=parameter_grid,
            scoring={
                "average_precision": "average_precision",
                "roc_auc": "roc_auc",
            },
            refit="average_precision",
            cv=cross_validation,
            n_jobs=1,
        )
        search.fit(training_features, training_target)

        cv_results = search.cv_results_
        for index, parameters in enumerate(cv_results["params"]):
            trial_result = {
                "algorithm": name,
                "trial": index + 1,
                "parameters": parameters,
                "cv_mean_pr_auc": float(cv_results["mean_test_average_precision"][index]),
                "cv_std_pr_auc": float(cv_results["std_test_average_precision"][index]),
                "cv_mean_roc_auc": float(cv_results["mean_test_roc_auc"][index]),
            }
            trials.append(trial_result)

            if mlflow is not None:
                with mlflow.start_run(run_name=f"{name}_trial_{index + 1}"):
                    mlflow.log_param("algorithm", name)
                    mlflow.log_params({
                        key.replace("classifier__", ""): value
                        for key, value in parameters.items()
                    })
                    mlflow.log_metrics({
                        "cv_mean_pr_auc": trial_result["cv_mean_pr_auc"],
                        "cv_std_pr_auc": trial_result["cv_std_pr_auc"],
                        "cv_mean_roc_auc": trial_result["cv_mean_roc_auc"],
                    })

        comparisons.append({
            "algorithm": name,
            "best_cv_pr_auc": float(search.best_score_),
            "best_params": search.best_params_,
        })
        print(f"{name}: mean CV average precision = {search.best_score_:.4f}")

        if best_result is None or search.best_score_ > best_result["cv_score"]:
            best_result = {
                "name": name,
                "model": search.best_estimator_,
                "cv_score": float(search.best_score_),
                "parameters": search.best_params_,
            }

    fitted_model = best_result["model"]
    probabilities = fitted_model.predict_proba(test_features)[:, 1]
    predicted_labels = (probabilities >= 0.50).astype(int)

    top_count = max(1, int(np.ceil(0.20 * len(test))))
    top_rows = np.argsort(-probabilities)[:top_count]
    overall_purchase_rate = float(test_target.mean())
    top_purchase_rate = float(test_target.iloc[top_rows].mean())

    metrics = {
        "selected_algorithm": best_result["name"],
        "selected_hyperparameters": best_result["parameters"],
        "selected_cv_pr_auc": best_result["cv_score"],
        "holdout_average_precision": float(
            average_precision_score(test_target, probabilities)
        ),
        "holdout_roc_auc": float(roc_auc_score(test_target, probabilities)),
        "holdout_precision_at_0_5": float(
            precision_score(test_target, predicted_labels, zero_division=0)
        ),
        "holdout_recall_at_0_5": float(
            recall_score(test_target, predicted_labels, zero_division=0)
        ),
        "holdout_f1_at_0_5": float(
            f1_score(test_target, predicted_labels, zero_division=0)
        ),
        "holdout_accuracy_at_0_5": float(
            accuracy_score(test_target, predicted_labels)
        ),
        "test_positive_rate": overall_purchase_rate,
        "top_20pct_positive_rate": top_purchase_rate,
        "top_20pct_lift": (
            top_purchase_rate / overall_purchase_rate
            if overall_purchase_rate > 0 else None
        ),
        "confusion_matrix": confusion_matrix(
            test_target, predicted_labels
        ).tolist(),
        "n_train": len(train),
        "n_test": len(test),
        "decision_threshold": 0.50,
        "tracking_mode": "mlflow_and_json" if mlflow else "json",
        "cv_results_by_model": comparisons,
    }

    metadata = {
        "features": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
        "categories": {
            column: sorted(str(value) for value in train[column].dropna().unique())
            for column in CATEGORICAL_COLUMNS
        },
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": best_result["name"],
        "positive_class": 1,
    }

    joblib.dump(fitted_model, model_dir / "best_model.joblib", compress=3)
    (model_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=json_value) + "\n",
        encoding="utf-8",
    )
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=json_value) + "\n",
        encoding="utf-8",
    )
    (report_dir / "experiments.json").write_text(
        json.dumps(trials, indent=2, default=json_value) + "\n",
        encoding="utf-8",
    )

    print(f"Selected model: {best_result['name']}")
    print(f"Test ROC-AUC: {metrics['holdout_roc_auc']:.3f}")
    print(f"Test average precision: {metrics['holdout_average_precision']:.3f}")
    print(classification_report(test_target, predicted_labels, zero_division=0))
    return fitted_model, metrics, trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--tracking-dir", type=Path, default=ROOT / "mlruns")
    args = parser.parse_args()
    experiment(
        args.input_dir / "train.csv",
        args.input_dir / "test.csv",
        args.model_dir,
        args.reports_dir,
        args.tracking_dir,
    )


if __name__ == "__main__":
    main()