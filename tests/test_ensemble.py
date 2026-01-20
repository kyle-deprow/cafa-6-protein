"""Tests for Tier 0 ensemble (GOA baseline + kNN predictions).

TDD tests for blending multiple prediction sources.
"""

import pandas as pd
import pytest

from cafa_6_protein.models.ensemble import (
    blend_predictions,
    merge_predictions,
    Tier0Ensemble,
)


class TestBlendPredictions:
    """Tests for blending two prediction DataFrames."""

    def test_blend_basic(self) -> None:
        """Test basic blending of two prediction sources."""
        pred1 = pd.DataFrame({
            "protein_id": ["P12345", "P12345"],
            "go_term": ["GO:0005524", "GO:0005737"],
            "score": [1.0, 0.8],
        })
        pred2 = pd.DataFrame({
            "protein_id": ["P12345", "P12345"],
            "go_term": ["GO:0005524", "GO:0003677"],
            "score": [0.6, 0.9],
        })

        blended = blend_predictions(pred1, pred2, weight1=0.4, weight2=0.6)

        # GO:0005524 should be weighted average: 0.4*1.0 + 0.6*0.6 = 0.76
        go_5524 = blended[
            (blended["protein_id"] == "P12345") & 
            (blended["go_term"] == "GO:0005524")
        ]
        assert len(go_5524) == 1
        assert go_5524.iloc[0]["score"] == pytest.approx(0.76)

    def test_blend_non_overlapping_terms(self) -> None:
        """Test blending when terms don't overlap."""
        pred1 = pd.DataFrame({
            "protein_id": ["P12345"],
            "go_term": ["GO:0005524"],
            "score": [1.0],
        })
        pred2 = pd.DataFrame({
            "protein_id": ["P12345"],
            "go_term": ["GO:0003677"],
            "score": [0.8],
        })

        blended = blend_predictions(pred1, pred2, weight1=0.5, weight2=0.5)

        # Both terms should appear, with scores weighted by their presence
        # GO:0005524: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        # GO:0003677: 0.5 * 0.0 + 0.5 * 0.8 = 0.4
        terms = set(blended["go_term"])
        assert "GO:0005524" in terms
        assert "GO:0003677" in terms

    def test_blend_empty_first(self) -> None:
        """Test blending when first prediction is empty."""
        pred1 = pd.DataFrame(columns=["protein_id", "go_term", "score"])
        pred2 = pd.DataFrame({
            "protein_id": ["P12345"],
            "go_term": ["GO:0005524"],
            "score": [0.8],
        })

        blended = blend_predictions(pred1, pred2, weight1=0.4, weight2=0.6)

        assert len(blended) == 1
        # Score should be weighted by presence: 0.6 * 0.8 = 0.48
        assert blended.iloc[0]["score"] == pytest.approx(0.48)

    def test_blend_multiple_proteins(self) -> None:
        """Test blending with multiple proteins."""
        pred1 = pd.DataFrame({
            "protein_id": ["P12345", "P67890"],
            "go_term": ["GO:0005524", "GO:0003677"],
            "score": [1.0, 0.9],
        })
        pred2 = pd.DataFrame({
            "protein_id": ["P12345", "P67890"],
            "go_term": ["GO:0005524", "GO:0003677"],
            "score": [0.5, 0.7],
        })

        blended = blend_predictions(pred1, pred2, weight1=0.5, weight2=0.5)

        assert len(blended) == 2
        proteins = set(blended["protein_id"])
        assert "P12345" in proteins
        assert "P67890" in proteins


class TestMergePredictions:
    """Tests for merging multiple prediction DataFrames."""

    def test_merge_basic(self) -> None:
        """Test merging predictions taking max score."""
        pred1 = pd.DataFrame({
            "protein_id": ["P12345"],
            "go_term": ["GO:0005524"],
            "score": [0.6],
        })
        pred2 = pd.DataFrame({
            "protein_id": ["P12345"],
            "go_term": ["GO:0005524"],
            "score": [0.8],
        })

        merged = merge_predictions([pred1, pred2], strategy="max")

        assert len(merged) == 1
        assert merged.iloc[0]["score"] == pytest.approx(0.8)

    def test_merge_mean_strategy(self) -> None:
        """Test merging with mean strategy."""
        pred1 = pd.DataFrame({
            "protein_id": ["P12345"],
            "go_term": ["GO:0005524"],
            "score": [0.6],
        })
        pred2 = pd.DataFrame({
            "protein_id": ["P12345"],
            "go_term": ["GO:0005524"],
            "score": [0.8],
        })

        merged = merge_predictions([pred1, pred2], strategy="mean")

        assert len(merged) == 1
        assert merged.iloc[0]["score"] == pytest.approx(0.7)

    def test_merge_empty_list(self) -> None:
        """Test merging empty list of predictions."""
        merged = merge_predictions([], strategy="max")

        assert len(merged) == 0


class TestTier0Ensemble:
    """Tests for the Tier 0 ensemble class."""

    def test_tier0_goa_only(self) -> None:
        """Test Tier 0 with only GOA predictions."""
        goa_preds = pd.DataFrame({
            "protein_id": ["P12345", "P12345"],
            "go_term": ["GO:0005524", "GO:0005737"],
            "score": [1.0, 1.0],
        })

        ensemble = Tier0Ensemble(goa_weight=1.0, knn_weight=0.0)
        result = ensemble.predict(goa_predictions=goa_preds, knn_predictions=None)

        assert len(result) == 2
        assert all(result["score"] == 1.0)

    def test_tier0_knn_only(self) -> None:
        """Test Tier 0 with only kNN predictions."""
        knn_preds = pd.DataFrame({
            "protein_id": ["P12345"],
            "go_term": ["GO:0005524"],
            "score": [0.8],
        })

        ensemble = Tier0Ensemble(goa_weight=0.0, knn_weight=1.0)
        result = ensemble.predict(goa_predictions=None, knn_predictions=knn_preds)

        assert len(result) == 1
        assert result.iloc[0]["score"] == pytest.approx(0.8)

    def test_tier0_combined(self) -> None:
        """Test Tier 0 with both GOA and kNN."""
        goa_preds = pd.DataFrame({
            "protein_id": ["P12345"],
            "go_term": ["GO:0005524"],
            "score": [1.0],
        })
        knn_preds = pd.DataFrame({
            "protein_id": ["P12345", "P12345"],
            "go_term": ["GO:0005524", "GO:0003677"],
            "score": [0.5, 0.9],
        })

        ensemble = Tier0Ensemble(goa_weight=0.4, knn_weight=0.6)
        result = ensemble.predict(goa_predictions=goa_preds, knn_predictions=knn_preds)

        # Should have both terms
        terms = set(result["go_term"])
        assert "GO:0005524" in terms
        assert "GO:0003677" in terms

    def test_tier0_normalizes_weights(self) -> None:
        """Test that weights are normalized if they don't sum to 1."""
        ensemble = Tier0Ensemble(goa_weight=2.0, knn_weight=3.0)
        
        # Weights should be normalized to 0.4 and 0.6
        assert ensemble.goa_weight == pytest.approx(0.4)
        assert ensemble.knn_weight == pytest.approx(0.6)

    def test_tier0_scores_in_valid_range(self) -> None:
        """Test that all output scores are in (0, 1]."""
        goa_preds = pd.DataFrame({
            "protein_id": ["P12345"],
            "go_term": ["GO:0005524"],
            "score": [1.0],
        })
        knn_preds = pd.DataFrame({
            "protein_id": ["P12345"],
            "go_term": ["GO:0005524"],
            "score": [0.8],
        })

        ensemble = Tier0Ensemble(goa_weight=0.5, knn_weight=0.5)
        result = ensemble.predict(goa_predictions=goa_preds, knn_predictions=knn_preds)

        assert all(result["score"] > 0)
        assert all(result["score"] <= 1.0)
