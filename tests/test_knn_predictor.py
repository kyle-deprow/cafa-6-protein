"""Tests for kNN-based GO term prediction.

TDD tests for nearest neighbor prediction using protein embeddings.
"""

import numpy as np
import pandas as pd
import pytest

from cafa_6_protein.models.knn import (
    KNNPredictor,
    distance_weighted_vote,
    aggregate_neighbor_annotations,
)


class TestDistanceWeightedVote:
    """Tests for distance-weighted voting function."""

    def test_distance_weighted_vote_basic(self) -> None:
        """Test basic distance-weighted voting."""
        # Closer neighbors should have more influence
        distances = np.array([0.1, 0.5, 1.0])
        annotations = [
            {"GO:0005524": 1.0, "GO:0005737": 1.0},  # neighbor 1 (closest)
            {"GO:0005524": 1.0},  # neighbor 2
            {"GO:0003677": 1.0},  # neighbor 3 (farthest)
        ]

        scores = distance_weighted_vote(distances, annotations)

        # GO:0005524 should have highest score (2 neighbors, one very close)
        assert "GO:0005524" in scores
        assert "GO:0005737" in scores
        assert "GO:0003677" in scores
        assert scores["GO:0005524"] > scores["GO:0003677"]  # more neighbors + closer

    def test_distance_weighted_vote_zero_distance(self) -> None:
        """Test handling of zero distance (exact match)."""
        distances = np.array([0.0, 0.5])
        annotations = [
            {"GO:0005524": 1.0},
            {"GO:0005737": 1.0},
        ]

        scores = distance_weighted_vote(distances, annotations)

        # Zero distance should give very high weight (using 1/(1+d))
        assert scores["GO:0005524"] > scores["GO:0005737"]

    def test_distance_weighted_vote_empty_annotations(self) -> None:
        """Test with empty annotations."""
        distances = np.array([0.1])
        annotations = [{}]

        scores = distance_weighted_vote(distances, annotations)

        assert scores == {}

    def test_distance_weighted_vote_normalized(self) -> None:
        """Test that scores are normalized to [0, 1]."""
        distances = np.array([0.1, 0.2, 0.3])
        annotations = [
            {"GO:0005524": 1.0},
            {"GO:0005524": 1.0},
            {"GO:0005524": 1.0},
        ]

        scores = distance_weighted_vote(distances, annotations, normalize=True)

        assert 0 < scores["GO:0005524"] <= 1.0


class TestAggregateNeighborAnnotations:
    """Tests for aggregating annotations from neighbor proteins."""

    def test_aggregate_basic(self) -> None:
        """Test basic annotation aggregation."""
        neighbor_ids = ["P12345", "P67890"]
        annotations_df = pd.DataFrame({
            "protein_id": ["P12345", "P12345", "P67890", "Q99999"],
            "go_term": ["GO:0005524", "GO:0005737", "GO:0005524", "GO:0003677"],
        })

        result = aggregate_neighbor_annotations(neighbor_ids, annotations_df)

        assert len(result) == 2
        assert "GO:0005524" in result[0]  # P12345
        assert "GO:0005737" in result[0]  # P12345
        assert "GO:0005524" in result[1]  # P67890
        assert "GO:0003677" not in result[1]  # Q99999 not a neighbor

    def test_aggregate_missing_neighbor(self) -> None:
        """Test handling of neighbors not in annotations."""
        neighbor_ids = ["P12345", "MISSING"]
        annotations_df = pd.DataFrame({
            "protein_id": ["P12345"],
            "go_term": ["GO:0005524"],
        })

        result = aggregate_neighbor_annotations(neighbor_ids, annotations_df)

        assert len(result) == 2
        assert "GO:0005524" in result[0]
        assert result[1] == {}  # MISSING has no annotations


class TestKNNPredictor:
    """Tests for the KNN predictor class."""

    @pytest.fixture
    def sample_embeddings(self) -> dict[str, np.ndarray]:
        """Sample protein embeddings for testing."""
        np.random.seed(42)
        return {
            "P12345": np.random.randn(64).astype(np.float32),
            "P67890": np.random.randn(64).astype(np.float32),
            "Q11111": np.random.randn(64).astype(np.float32),
            "Q22222": np.random.randn(64).astype(np.float32),
        }

    @pytest.fixture
    def sample_train_annotations(self) -> pd.DataFrame:
        """Sample training annotations."""
        return pd.DataFrame({
            "protein_id": ["P12345", "P12345", "P67890", "Q11111", "Q11111"],
            "go_term": ["GO:0005524", "GO:0005737", "GO:0005524", "GO:0003677", "GO:0005737"],
        })

    def test_knn_predictor_init(
        self,
        sample_embeddings: dict[str, np.ndarray],
        sample_train_annotations: pd.DataFrame,
    ) -> None:
        """Test KNN predictor initialization."""
        predictor = KNNPredictor(k=3)
        predictor.fit(sample_embeddings, sample_train_annotations)

        assert predictor.k == 3
        assert predictor._index is not None
        assert len(predictor._protein_ids) == 4

    def test_knn_predictor_predict_single(
        self,
        sample_embeddings: dict[str, np.ndarray],
        sample_train_annotations: pd.DataFrame,
    ) -> None:
        """Test prediction for a single protein."""
        predictor = KNNPredictor(k=2)
        predictor.fit(sample_embeddings, sample_train_annotations)

        # Create a test embedding similar to P12345
        test_embedding = sample_embeddings["P12345"] + 0.01 * np.random.randn(64).astype(np.float32)
        
        predictions = predictor.predict_one("TEST001", test_embedding)

        assert isinstance(predictions, pd.DataFrame)
        assert "protein_id" in predictions.columns
        assert "go_term" in predictions.columns
        assert "score" in predictions.columns
        assert len(predictions) > 0
        assert all(predictions["protein_id"] == "TEST001")

    def test_knn_predictor_predict_batch(
        self,
        sample_embeddings: dict[str, np.ndarray],
        sample_train_annotations: pd.DataFrame,
    ) -> None:
        """Test prediction for multiple proteins."""
        predictor = KNNPredictor(k=2)
        predictor.fit(sample_embeddings, sample_train_annotations)

        test_embeddings = {
            "TEST001": sample_embeddings["P12345"] + 0.01 * np.random.randn(64).astype(np.float32),
            "TEST002": sample_embeddings["P67890"] + 0.01 * np.random.randn(64).astype(np.float32),
        }
        
        predictions = predictor.predict(test_embeddings)

        assert len(predictions["protein_id"].unique()) == 2
        assert "TEST001" in predictions["protein_id"].values
        assert "TEST002" in predictions["protein_id"].values

    def test_knn_predictor_k_larger_than_training(
        self,
        sample_embeddings: dict[str, np.ndarray],
        sample_train_annotations: pd.DataFrame,
    ) -> None:
        """Test handling when k is larger than training set."""
        predictor = KNNPredictor(k=100)  # Much larger than 4 training proteins
        predictor.fit(sample_embeddings, sample_train_annotations)

        test_embedding = sample_embeddings["P12345"].copy()
        predictions = predictor.predict_one("TEST001", test_embedding)

        # Should still work, using all available neighbors
        assert len(predictions) > 0

    def test_knn_predictor_scores_in_valid_range(
        self,
        sample_embeddings: dict[str, np.ndarray],
        sample_train_annotations: pd.DataFrame,
    ) -> None:
        """Test that all prediction scores are in (0, 1]."""
        predictor = KNNPredictor(k=3)
        predictor.fit(sample_embeddings, sample_train_annotations)

        test_embedding = sample_embeddings["P12345"] + 0.1 * np.random.randn(64).astype(np.float32)
        predictions = predictor.predict_one("TEST001", test_embedding)

        assert all(predictions["score"] > 0)
        assert all(predictions["score"] <= 1.0)


class TestKNNPredictorWithoutFaiss:
    """Tests for KNN predictor using sklearn fallback when FAISS not available."""

    def test_sklearn_fallback_works(self) -> None:
        """Test that sklearn fallback produces valid results."""
        np.random.seed(42)
        embeddings = {
            "P12345": np.random.randn(32).astype(np.float32),
            "P67890": np.random.randn(32).astype(np.float32),
        }
        annotations = pd.DataFrame({
            "protein_id": ["P12345", "P67890"],
            "go_term": ["GO:0005524", "GO:0003677"],
        })

        predictor = KNNPredictor(k=1, use_faiss=False)
        predictor.fit(embeddings, annotations)

        test_emb = embeddings["P12345"] + 0.01 * np.random.randn(32).astype(np.float32)
        predictions = predictor.predict_one("TEST", test_emb)

        assert len(predictions) > 0
        assert "GO:0005524" in predictions["go_term"].values or "GO:0003677" in predictions["go_term"].values
