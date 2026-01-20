"""Tests for the retrieval-augmented predictor.

Tests Stage 4 (Score Aggregation) of the solution plan.
"""

import numpy as np
import pandas as pd
import pytest

from cafa_6_protein.models.retrieval import (
    AggregationConfig,
    NeighborEvidence,
    RetrievalAugmentedPredictor,
    aggregate_scores,
    convert_distance_to_similarity,
    create_retrieval_predictor,
)


class TestConvertDistanceToSimilarity:
    """Tests for distance to similarity conversion."""

    def test_inverse_method(self) -> None:
        """Test inverse conversion: 1 / (1 + distance)."""
        distances = np.array([0.0, 1.0, 2.0, 10.0])
        similarities = convert_distance_to_similarity(distances, method="inverse")

        assert similarities[0] == pytest.approx(1.0)  # distance 0 -> sim 1
        assert similarities[1] == pytest.approx(0.5)  # distance 1 -> sim 0.5
        assert similarities[2] == pytest.approx(1 / 3)  # distance 2 -> sim 0.33
        assert similarities[3] == pytest.approx(1 / 11)  # distance 10 -> sim ~0.09

    def test_exp_method(self) -> None:
        """Test exponential conversion: exp(-distance)."""
        distances = np.array([0.0, 1.0, 2.0])
        similarities = convert_distance_to_similarity(distances, method="exp")

        assert similarities[0] == pytest.approx(1.0)  # distance 0 -> sim 1
        assert similarities[1] == pytest.approx(np.exp(-1))
        assert similarities[2] == pytest.approx(np.exp(-2))

    def test_unknown_method_raises(self) -> None:
        """Test that unknown method raises ValueError."""
        distances = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="Unknown conversion method"):
            convert_distance_to_similarity(distances, method="unknown")


class TestAggregateScores:
    """Tests for score aggregation from neighbor evidence."""

    def test_empty_evidence(self) -> None:
        """Test aggregation with no evidence."""
        result = aggregate_scores([])
        assert result == {}

    def test_single_neighbor_label_only(self) -> None:
        """Test aggregation with one neighbor, label-based only."""
        evidence = NeighborEvidence(
            protein_id="P1",
            similarity=1.0,
            go_terms={"GO:0001", "GO:0002"},
        )
        config = AggregationConfig(alpha=1.0)  # Only labels
        result = aggregate_scores([evidence], config)

        # With normalization, max score should be 1.0
        assert len(result) == 2
        assert result["GO:0001"] == pytest.approx(1.0)
        assert result["GO:0002"] == pytest.approx(1.0)

    def test_single_neighbor_literature_only(self) -> None:
        """Test aggregation with one neighbor, literature-based only."""
        evidence = NeighborEvidence(
            protein_id="P1",
            similarity=1.0,
            literature_terms={"GO:0003", "GO:0004"},
        )
        config = AggregationConfig(alpha=0.0, literature_discount=1.0)  # Only literature
        result = aggregate_scores([evidence], config)

        assert len(result) == 2
        assert result["GO:0003"] == pytest.approx(1.0)
        assert result["GO:0004"] == pytest.approx(1.0)

    def test_mixed_labels_and_literature(self) -> None:
        """Test aggregation combining labels and literature."""
        evidence = NeighborEvidence(
            protein_id="P1",
            similarity=1.0,
            go_terms={"GO:0001"},
            literature_terms={"GO:0002"},
        )
        config = AggregationConfig(
            alpha=0.7,
            literature_discount=0.5,
            normalize_scores=False,
        )
        result = aggregate_scores([evidence], config)

        # GO:0001: 0.7 * 1.0 = 0.7
        # GO:0002: 0.3 * 1.0 * 0.5 = 0.15
        assert result["GO:0001"] == pytest.approx(0.7)
        assert result["GO:0002"] == pytest.approx(0.15)

    def test_multiple_neighbors_additive(self) -> None:
        """Test that scores from multiple neighbors add up."""
        evidence1 = NeighborEvidence(
            protein_id="P1",
            similarity=1.0,
            go_terms={"GO:0001"},
        )
        evidence2 = NeighborEvidence(
            protein_id="P2",
            similarity=0.5,
            go_terms={"GO:0001"},
        )
        config = AggregationConfig(alpha=1.0, normalize_scores=False)
        result = aggregate_scores([evidence1, evidence2], config)

        # GO:0001: 1.0 * 1.0 + 0.5 * 1.0 = 1.5
        assert result["GO:0001"] == pytest.approx(1.5)

    def test_threshold_applied(self) -> None:
        """Test that scores below threshold are filtered."""
        evidence = NeighborEvidence(
            protein_id="P1",
            similarity=0.01,  # Low similarity
            go_terms={"GO:0001"},
        )
        config = AggregationConfig(
            alpha=1.0,
            min_score_threshold=0.1,
            normalize_scores=False,
        )
        result = aggregate_scores([evidence], config)

        # Score is 0.01, below threshold of 0.1
        assert "GO:0001" not in result

    def test_normalization(self) -> None:
        """Test that normalization scales to [0, 1]."""
        evidence1 = NeighborEvidence(
            protein_id="P1",
            similarity=1.0,
            go_terms={"GO:0001"},
        )
        evidence2 = NeighborEvidence(
            protein_id="P2",
            similarity=0.5,
            go_terms={"GO:0001", "GO:0002"},
        )
        config = AggregationConfig(alpha=1.0, normalize_scores=True)
        result = aggregate_scores([evidence1, evidence2], config)

        # GO:0001: 1.0 + 0.5 = 1.5 -> normalized to 1.0
        # GO:0002: 0.5 -> normalized to 0.5/1.5 = 0.333
        assert result["GO:0001"] == pytest.approx(1.0)
        assert result["GO:0002"] == pytest.approx(1 / 3)


class TestRetrievalAugmentedPredictor:
    """Tests for the RetrievalAugmentedPredictor class."""

    @pytest.fixture
    def sample_embeddings(self) -> dict[str, np.ndarray]:
        """Create sample training embeddings."""
        return {
            "P1": np.array([1.0, 0.0, 0.0, 0.0]),
            "P2": np.array([0.9, 0.1, 0.0, 0.0]),
            "P3": np.array([0.0, 1.0, 0.0, 0.0]),
            "P4": np.array([0.0, 0.0, 1.0, 0.0]),
        }

    @pytest.fixture
    def sample_annotations(self) -> pd.DataFrame:
        """Create sample training annotations."""
        return pd.DataFrame([
            {"protein_id": "P1", "go_term": "GO:0001"},
            {"protein_id": "P1", "go_term": "GO:0002"},
            {"protein_id": "P2", "go_term": "GO:0001"},
            {"protein_id": "P2", "go_term": "GO:0003"},
            {"protein_id": "P3", "go_term": "GO:0004"},
            {"protein_id": "P4", "go_term": "GO:0005"},
        ])

    def test_fit(
        self,
        sample_embeddings: dict[str, np.ndarray],
        sample_annotations: pd.DataFrame,
    ) -> None:
        """Test fitting the predictor."""
        predictor = RetrievalAugmentedPredictor(k=2, use_faiss=False)
        predictor.fit(sample_embeddings, sample_annotations)

        assert len(predictor._protein_ids) == 4
        assert predictor._embedding_dim == 4
        assert "P1" in predictor._annotations_by_protein
        assert "GO:0001" in predictor._annotations_by_protein["P1"]

    def test_predict_one_similar_query(
        self,
        sample_embeddings: dict[str, np.ndarray],
        sample_annotations: pd.DataFrame,
    ) -> None:
        """Test predicting for a query similar to P1/P2."""
        predictor = RetrievalAugmentedPredictor(k=2, use_faiss=False)
        predictor.fit(sample_embeddings, sample_annotations)

        # Query similar to P1 and P2
        query = np.array([0.95, 0.05, 0.0, 0.0])
        result = predictor.predict_one("test_protein", query)

        assert not result.empty
        assert set(result.columns) == {"protein_id", "go_term", "score"}
        assert all(result["protein_id"] == "test_protein")

        # Should have terms from P1 and P2
        terms = set(result["go_term"])
        assert "GO:0001" in terms  # Common to P1 and P2
        assert "GO:0002" in terms or "GO:0003" in terms  # From one of them

    def test_predict_multiple(
        self,
        sample_embeddings: dict[str, np.ndarray],
        sample_annotations: pd.DataFrame,
    ) -> None:
        """Test predicting for multiple proteins."""
        predictor = RetrievalAugmentedPredictor(k=2, use_faiss=False)
        predictor.fit(sample_embeddings, sample_annotations)

        test_embeddings = {
            "Q1": np.array([1.0, 0.0, 0.0, 0.0]),
            "Q2": np.array([0.0, 1.0, 0.0, 0.0]),
        }
        result = predictor.predict(test_embeddings)

        assert not result.empty
        protein_ids = set(result["protein_id"])
        assert protein_ids == {"Q1", "Q2"}

    def test_get_params(
        self,
        sample_embeddings: dict[str, np.ndarray],
        sample_annotations: pd.DataFrame,
    ) -> None:
        """Test getting predictor parameters."""
        config = AggregationConfig(alpha=0.8, literature_discount=0.3)
        predictor = RetrievalAugmentedPredictor(k=10, config=config, use_faiss=False)
        predictor.fit(sample_embeddings, sample_annotations)

        params = predictor.get_params()
        assert params.k == 10
        assert params.alpha == 0.8
        assert params.literature_discount == 0.3
        assert params.has_literature is False
        assert params.has_ontology is False

    def test_get_neighbor_explanation(
        self,
        sample_embeddings: dict[str, np.ndarray],
        sample_annotations: pd.DataFrame,
    ) -> None:
        """Test getting interpretable neighbor explanations."""
        predictor = RetrievalAugmentedPredictor(k=3, use_faiss=False)
        predictor.fit(sample_embeddings, sample_annotations)

        query = np.array([1.0, 0.0, 0.0, 0.0])
        explanations = predictor.get_neighbor_explanation(query, top_n=2)

        assert len(explanations) == 2
        assert hasattr(explanations[0], "protein_id")
        assert hasattr(explanations[0], "similarity")
        assert hasattr(explanations[0], "num_go_terms")


class TestCreateRetrievalPredictor:
    """Tests for the factory function."""

    def test_basic_creation(self) -> None:
        """Test creating a basic predictor without optional components."""
        embeddings = {
            "P1": np.array([1.0, 0.0]),
            "P2": np.array([0.0, 1.0]),
        }
        annotations = pd.DataFrame([
            {"protein_id": "P1", "go_term": "GO:0001"},
            {"protein_id": "P2", "go_term": "GO:0002"},
        ])

        predictor = create_retrieval_predictor(
            train_embeddings=embeddings,
            annotations=annotations,
            k=2,
            alpha=0.6,
        )

        assert predictor.k == 2
        assert predictor.config.alpha == 0.6
        assert len(predictor._protein_ids) == 2


class TestIntegrationWithOntology:
    """Integration tests with ontology propagation."""

    @pytest.fixture
    def simple_ontology(self) -> "nx.DiGraph":
        """Create a simple test ontology."""
        import networkx as nx

        # Create GO DAG: GO:0001 -> GO:0002 -> GO:0003 (root)
        # (edges point child -> parent in obonet convention)
        G = nx.DiGraph()
        G.add_edge("GO:0001", "GO:0002")
        G.add_edge("GO:0002", "GO:0003")
        return G

    def test_ancestor_propagation(
        self,
        simple_ontology: "nx.DiGraph",
    ) -> None:
        """Test that scores are propagated to ancestors."""
        from cafa_6_protein.models.retrieval import propagate_to_ancestors

        term_scores = {"GO:0001": 0.8}
        propagated = propagate_to_ancestors(term_scores, simple_ontology)

        # Should have GO:0001, GO:0002, GO:0003 all with score 0.8
        assert propagated["GO:0001"] == pytest.approx(0.8)
        assert propagated["GO:0002"] == pytest.approx(0.8)
        assert propagated["GO:0003"] == pytest.approx(0.8)

    def test_max_propagation(
        self,
        simple_ontology: "nx.DiGraph",
    ) -> None:
        """Test that max score is used when multiple children."""
        from cafa_6_protein.models.retrieval import propagate_to_ancestors

        term_scores = {"GO:0001": 0.8, "GO:0002": 0.5}
        propagated = propagate_to_ancestors(term_scores, simple_ontology)

        # GO:0002 should get max(0.8, 0.5) = 0.8 from GO:0001's propagation
        assert propagated["GO:0001"] == pytest.approx(0.8)
        assert propagated["GO:0002"] == pytest.approx(0.8)
        assert propagated["GO:0003"] == pytest.approx(0.8)
