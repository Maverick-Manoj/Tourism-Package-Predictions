"""Tests for the decision-threshold helpers and the candidate model list."""

import numpy as np

from src.train_model import best_f1_threshold, candidate_models, threshold_sweep


def test_threshold_separates_clean_scores():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.10, 0.20, 0.30, 0.60, 0.70, 0.90])

    threshold, f1 = best_f1_threshold(labels, scores)

    assert f1 == 1.0
    assert 0.30 < threshold <= 0.60


def test_threshold_stays_inside_allowed_range():
    labels = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])

    threshold, _ = best_f1_threshold(labels, scores)

    assert 0.05 <= threshold <= 0.95


def test_sweep_reports_workload_and_recall():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.2, 0.4, 0.6, 0.8])

    rows = threshold_sweep(labels, scores)

    assert rows
    for row in rows:
        assert 0.0 <= row["recall"] <= 1.0
        assert 0.0 <= row["customers_flagged_pct"] <= 100.0


def test_required_algorithms_are_compared():
    names = set(candidate_models())

    assert {
        "DecisionTree", "Bagging", "RandomForest", "AdaBoost",
        "GradientBoosting",
    } <= names