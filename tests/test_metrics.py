"""Tests for evaluation metrics."""

import pandas as pd
import pytest

from cafa_6_protein.evaluation.metrics import (
    max_weighted_f1,
    weighted_f1_score,
    weighted_precision,
    weighted_recall,
)


class TestWeightedPrecision:
    """Tests for weighted precision metric."""

    def test_perfect_precision(self, sample_weights: dict[str, float]) -> None:
        """Test precision when all predictions are correct."""
        predictions = pd.DataFrame(
            {
                "protein_id": ["P12345", "P12345"],
                "go_term": ["GO:0005524", "GO:0016887"],
                "score": [0.9, 0.8],
            }
        )
        ground_truth = pd.DataFrame(
            {
                "protein_id": ["P12345", "P12345"],
                "go_term": ["GO:0005524", "GO:0016887"],
            }
        )

        precision = weighted_precision(predictions, ground_truth, sample_weights)
        assert precision == pytest.approx(1.0)

    def test_zero_precision(self, sample_weights: dict[str, float]) -> None:
        """Test precision when no predictions are correct."""
        predictions = pd.DataFrame(
            {
                "protein_id": ["P12345"],
                "go_term": ["GO:0005524"],
                "score": [0.9],
            }
        )
        ground_truth = pd.DataFrame(
            {
                "protein_id": ["P12345"],
                "go_term": ["GO:0016887"],
            }
        )

        precision = weighted_precision(predictions, ground_truth, sample_weights)
        assert precision == pytest.approx(0.0)

    def test_empty_predictions(self, sample_weights: dict[str, float]) -> None:
        """Test precision with empty predictions."""
        predictions = pd.DataFrame(columns=["protein_id", "go_term", "score"])
        ground_truth = pd.DataFrame(
            {
                "protein_id": ["P12345"],
                "go_term": ["GO:0005524"],
            }
        )

        precision = weighted_precision(predictions, ground_truth, sample_weights)
        assert precision == 0.0


class TestWeightedRecall:
    """Tests for weighted recall metric."""

    def test_perfect_recall(self, sample_weights: dict[str, float]) -> None:
        """Test recall when all ground truth is predicted."""
        predictions = pd.DataFrame(
            {
                "protein_id": ["P12345", "P12345"],
                "go_term": ["GO:0005524", "GO:0016887"],
                "score": [0.9, 0.8],
            }
        )
        ground_truth = pd.DataFrame(
            {
                "protein_id": ["P12345", "P12345"],
                "go_term": ["GO:0005524", "GO:0016887"],
            }
        )

        recall = weighted_recall(predictions, ground_truth, sample_weights)
        assert recall == pytest.approx(1.0)

    def test_zero_recall(self, sample_weights: dict[str, float]) -> None:
        """Test recall when nothing is predicted correctly."""
        predictions = pd.DataFrame(
            {
                "protein_id": ["P12345"],
                "go_term": ["GO:0005524"],
                "score": [0.9],
            }
        )
        ground_truth = pd.DataFrame(
            {
                "protein_id": ["P12345"],
                "go_term": ["GO:0016887"],
            }
        )

        recall = weighted_recall(predictions, ground_truth, sample_weights)
        assert recall == pytest.approx(0.0)


class TestWeightedF1:
    """Tests for weighted F1 score."""

    def test_perfect_f1(self, sample_weights: dict[str, float]) -> None:
        """Test F1 with perfect predictions."""
        predictions = pd.DataFrame(
            {
                "protein_id": ["P12345"],
                "go_term": ["GO:0005524"],
                "score": [0.9],
            }
        )
        ground_truth = pd.DataFrame(
            {
                "protein_id": ["P12345"],
                "go_term": ["GO:0005524"],
            }
        )

        f1 = weighted_f1_score(predictions, ground_truth, sample_weights, threshold=0.5)
        assert f1 == pytest.approx(1.0)

    def test_threshold_filtering(self, sample_weights: dict[str, float]) -> None:
        """Test that threshold properly filters predictions."""
        predictions = pd.DataFrame(
            {
                "protein_id": ["P12345", "P12345"],
                "go_term": ["GO:0005524", "GO:0016887"],
                "score": [0.9, 0.3],  # Second prediction below threshold
            }
        )
        ground_truth = pd.DataFrame(
            {
                "protein_id": ["P12345", "P12345"],
                "go_term": ["GO:0005524", "GO:0016887"],
            }
        )

        # With threshold 0.5, only GO:0005524 is predicted
        f1 = weighted_f1_score(predictions, ground_truth, sample_weights, threshold=0.5)
        assert 0 < f1 < 1.0  # Partial score


class TestMaxWeightedF1:
    """Tests for maximum F1 search."""

    def test_finds_best_threshold(self, sample_weights: dict[str, float]) -> None:
        """Test that max F1 finds an optimal threshold."""
        predictions = pd.DataFrame(
            {
                "protein_id": ["P12345"],
                "go_term": ["GO:0005524"],
                "score": [0.75],
            }
        )
        ground_truth = pd.DataFrame(
            {
                "protein_id": ["P12345"],
                "go_term": ["GO:0005524"],
            }
        )

        max_f1, best_threshold = max_weighted_f1(predictions, ground_truth, sample_weights)
        assert max_f1 == pytest.approx(1.0)
        assert best_threshold <= 0.75
