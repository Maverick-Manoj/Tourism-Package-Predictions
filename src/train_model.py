"""Compare classifiers, choose a decision threshold and save the best model.

Candidate algorithms: Decision Tree, Bagging, Random Forest, AdaBoost,
Gradient Boosting and (when installed) XGBoost. Every algorithm is tuned with
stratified cross-validation on the training split only.

The decision threshold is chosen from out-of-fold predictions on the training
split, so the held-out test split is used only to report results, never to
pick the model, its parameters or its threshold.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
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

try:
    from xgboost import XGBClassifier
except ImportError:  # the notebook still runs where XGBoost is not installed
    XGBClassifier = None

RANDOM_SEED = 42
REFERENCE_THRESHOLD = 0.50
SWEEP_THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.05), 2)


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


def candidate_models(positive_weight=1.0):
    """Return {name: (estimator, parameter grid)} for every algorithm tried.

    ``positive_weight`` is the negative-to-positive ratio in the training
    data. It is one of the values tried for XGBoost's ``scale_pos_weight``.
    """
    seed = RANDOM_SEED

    models = {
        "DecisionTree": (
            DecisionTreeClassifier(random_state=seed),
            {
                "classifier__max_depth": [6, 10, None],
                "classifier__min_samples_leaf": [2, 5],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Bagging": (
            BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=seed),
                random_state=seed,
                n_jobs=1,
            ),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_samples": [0.8, 1.0],
                "classifier__estimator__min_samples_leaf": [1, 2],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=seed, n_jobs=1),
            {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__class_weight": [None, "balanced_subsample"],
            },
        ),
        "AdaBoost": (
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(random_state=seed),
                random_state=seed,
            ),
            {
                "classifier__n_estimators": [100, 300],
                "classifier__learning_rate": [0.3, 1.0],
                "classifier__estimator__max_depth": [1, 3],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=seed),
            {
                "classifier__n_estimators": [150, 300],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5],
            },
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=seed,
                n_jobs=1,
                verbosity=0,
            ),
            {
                "classifier__n_estimators": [200, 400],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [4, 6],
                "classifier__scale_pos_weight": [1.0, round(positive_weight, 2)],
            },
        )

    return models


def best_f1_threshold(y_true, scores, low=0.05, high=0.95):
    """Return (threshold, f1) that maximises F1, kept inside [low, high]."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]
    total = precision + recall
    f1 = np.divide(
        2 * precision * recall, total, out=np.zeros_like(total), where=total > 0
    )
    best = int(np.argmax(f1))
    threshold = float(np.clip(np.round(thresholds[best], 2), low, high))
    return threshold, float(f1[best])


def scores_at(y_true, probabilities, threshold):
    """Precision, recall, F1 and workload at one probability cutoff."""
    labels = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, labels)),
        "customers_flagged_pct": float(100 * labels.mean()),
        "confusion_matrix": confusion_matrix(
            y_true, labels, labels=[0, 1]
        ).tolist(),
    }


def threshold_sweep(y_true, probabilities):
    """Show the precision/recall trade-off across cutoffs (reporting only)."""
    rows = []
    for threshold in SWEEP_THRESHOLDS:
        result = scores_at(y_true, probabilities, threshold)
        result.pop("confusion_matrix")
        result.pop("accuracy")
        rows.append(result)
    return rows


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

    positive_weight = float(
        (training_target == 0).sum() / max((training_target == 1).sum(), 1)
    )
    cross_validation = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=RANDOM_SEED
    )
    trials = []
    comparisons = []
    best_result = None

    for name, (classifier, parameter_grid) in candidate_models(
        positive_weight
    ).items():
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
                    mlflow.set_tags({"stage": "tuning_trial", "algorithm": name})
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

    # Choose the cutoff from out-of-fold training predictions (no test data).
    out_of_fold = cross_val_predict(
        fitted_model,
        training_features,
        training_target,
        cv=cross_validation,
        method="predict_proba",
    )[:, 1]
    tuned_threshold, oof_f1 = best_f1_threshold(training_target, out_of_fold)

    probabilities = fitted_model.predict_proba(test_features)[:, 1]
    at_reference = scores_at(test_target, probabilities, REFERENCE_THRESHOLD)
    at_tuned = scores_at(test_target, probabilities, tuned_threshold)

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
        "holdout_precision_at_0_5": at_reference["precision"],
        "holdout_recall_at_0_5": at_reference["recall"],
        "holdout_f1_at_0_5": at_reference["f1"],
        "holdout_accuracy_at_0_5": at_reference["accuracy"],
        "tuned_threshold": tuned_threshold,
        "threshold_selection": "max F1 on out-of-fold training predictions",
        "threshold_train_oof_f1": oof_f1,
        "holdout_precision_at_tuned": at_tuned["precision"],
        "holdout_recall_at_tuned": at_tuned["recall"],
        "holdout_f1_at_tuned": at_tuned["f1"],
        "holdout_accuracy_at_tuned": at_tuned["accuracy"],
        "customers_flagged_pct_at_tuned": at_tuned["customers_flagged_pct"],
        "confusion_matrix_at_tuned": at_tuned["confusion_matrix"],
        "test_positive_rate": overall_purchase_rate,
        "top_20pct_positive_rate": top_purchase_rate,
        "top_20pct_lift": (
            top_purchase_rate / overall_purchase_rate
            if overall_purchase_rate > 0 else None
        ),
        "confusion_matrix": at_reference["confusion_matrix"],
        "n_train": len(train),
        "n_test": len(test),
        "decision_threshold": REFERENCE_THRESHOLD,
        "tracking_mode": "mlflow_and_json" if mlflow else "json",
        "cv_results_by_model": comparisons,
        "threshold_sweep": threshold_sweep(test_target, probabilities),
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
        "algorithms_compared": [row["algorithm"] for row in comparisons],
        "recommended_threshold": tuned_threshold,
        "scikit_learn_version": sklearn.__version__,
        "positive_class": 1,
    }

    model_file = model_dir / "best_model.joblib"
    metadata_file = model_dir / "model_metadata.json"
    metrics_file = report_dir / "metrics.json"
    experiments_file = report_dir / "experiments.json"

    joblib.dump(fitted_model, model_file, compress=3)
    metadata_file.write_text(
        json.dumps(metadata, indent=2, default=json_value) + "\n",
        encoding="utf-8",
    )
    metrics_file.write_text(
        json.dumps(metrics, indent=2, default=json_value) + "\n",
        encoding="utf-8",
    )
    experiments_file.write_text(
        json.dumps(trials, indent=2, default=json_value) + "\n",
        encoding="utf-8",
    )

    if mlflow is not None:
        with mlflow.start_run(run_name="selected_model"):
            mlflow.set_tags({
                "stage": "final",
                "algorithm": best_result["name"],
            })
            mlflow.log_param("algorithm", best_result["name"])
            mlflow.log_params({
                key.replace("classifier__", ""): value
                for key, value in best_result["parameters"].items()
            })
            mlflow.log_param("tuned_threshold", tuned_threshold)
            mlflow.log_metrics({
                "cv_mean_pr_auc": metrics["selected_cv_pr_auc"],
                "holdout_average_precision": metrics["holdout_average_precision"],
                "holdout_roc_auc": metrics["holdout_roc_auc"],
                "holdout_f1_at_0_5": metrics["holdout_f1_at_0_5"],
                "holdout_precision_at_tuned": metrics["holdout_precision_at_tuned"],
                "holdout_recall_at_tuned": metrics["holdout_recall_at_tuned"],
                "holdout_f1_at_tuned": metrics["holdout_f1_at_tuned"],
                "top_20pct_lift": metrics["top_20pct_lift"],
            })
            for artifact in (model_file, metadata_file, metrics_file):
                mlflow.log_artifact(str(artifact))

    print(f"Selected model: {best_result['name']}")
    print(f"Tuned threshold (train out-of-fold F1): {tuned_threshold:.2f}")
    print(f"Test ROC-AUC: {metrics['holdout_roc_auc']:.3f}")
    print(f"Test average precision: {metrics['holdout_average_precision']:.3f}")
    print(f"Test report at threshold {tuned_threshold:.2f}:")
    print(classification_report(
        test_target,
        (probabilities >= tuned_threshold).astype(int),
        zero_division=0,
    ))
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