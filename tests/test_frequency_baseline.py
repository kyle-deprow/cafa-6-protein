"""Tests for frequency-based baseline predictor."""

import pandas as pd
import pytest

from cafa_6_protein.models.frequency import FrequencyBaseline


class TestFrequencyBaseline:
    """Tests for FrequencyBaseline class."""

    @pytest.fixture
    def train_annotations(self) -> pd.DataFrame:
        """Sample training annotations."""
        return pd.DataFrame({
            "protein_id": ["P1", "P1", "P2", "P2", "P2", "P3"],
            "go_term": ["GO:0000001", "GO:0000002", "GO:0000001", "GO:0000002", "GO:0000003", "GO:0000001"],
        })

    @pytest.fixture
    def ia_weights(self) -> pd.DataFrame:
        """Sample IA weights."""
        return pd.DataFrame({
            "go_term": ["GO:0000001", "GO:0000002", "GO:0000003"],
            "ia": [1.0, 3.0, 5.0],
        })

    def test_fit_computes_term_frequencies(self, train_annotations: pd.DataFrame) -> None:
        """Test that fit() computes term frequencies."""
        baseline = FrequencyBaseline()
        baseline.fit(train_annotations)
        
        # GO:0000001 appears 3 times out of 3 proteins
        assert baseline.term_frequencies["GO:0000001"] == 1.0
        # GO:0000002 appears 2 times out of 3 proteins
        assert baseline.term_frequencies["GO:0000002"] == pytest.approx(2/3)
        # GO:0000003 appears 1 time out of 3 proteins
        assert baseline.term_frequencies["GO:0000003"] == pytest.approx(1/3)

    def test_fit_with_ia_weights(self, train_annotations: pd.DataFrame, ia_weights: pd.DataFrame) -> None:
        """Test that fit() uses IA weights for scoring."""
        baseline = FrequencyBaseline()
        baseline.fit(train_annotations, ia_weights=ia_weights)
        
        # Scores should be frequency * sqrt(IA) (normalized)
        assert "GO:0000003" in baseline.term_scores
        # Higher IA terms should get boosted
        assert baseline.term_scores["GO:0000003"] > baseline.term_scores["GO:0000001"] / 2

    def test_predict_returns_same_scores_for_all_proteins(self, train_annotations: pd.DataFrame) -> None:
        """Test that predict() returns same scores for all proteins."""
        baseline = FrequencyBaseline()
        baseline.fit(train_annotations)
        
        test_proteins = ["T1", "T2"]
        predictions = baseline.predict(test_proteins)
        
        assert "protein_id" in predictions.columns
        assert "go_term" in predictions.columns
        assert "score" in predictions.columns
        
        # Both proteins should have same terms
        t1_terms = set(predictions[predictions["protein_id"] == "T1"]["go_term"])
        t2_terms = set(predictions[predictions["protein_id"] == "T2"]["go_term"])
        assert t1_terms == t2_terms

    def test_predict_with_top_k(self, train_annotations: pd.DataFrame) -> None:
        """Test that predict() respects top_k parameter."""
        baseline = FrequencyBaseline(top_k=2)
        baseline.fit(train_annotations)
        
        predictions = baseline.predict(["T1"])
        terms_for_t1 = predictions[predictions["protein_id"] == "T1"]
        assert len(terms_for_t1) == 2

    def test_predict_before_fit_raises(self) -> None:
        """Test that predict() before fit() raises ValueError."""
        baseline = FrequencyBaseline()
        with pytest.raises(ValueError, match="must call fit"):
            baseline.predict(["T1"])

    def test_scores_are_normalized(self, train_annotations: pd.DataFrame) -> None:
        """Test that scores are in (0, 1] range."""
        baseline = FrequencyBaseline()
        baseline.fit(train_annotations)
        
        predictions = baseline.predict(["T1"])
        assert all(predictions["score"] > 0)
        assert all(predictions["score"] <= 1)

    def test_fit_filters_low_frequency_terms(self, train_annotations: pd.DataFrame) -> None:
        """Test that terms below min_frequency are excluded."""
        baseline = FrequencyBaseline(min_frequency=0.5)
        baseline.fit(train_annotations)
        
        # Only GO:0000001 (1.0) and GO:0000002 (0.67) should pass
        predictions = baseline.predict(["T1"])
        terms = set(predictions["go_term"])
        assert "GO:0000001" in terms
        assert "GO:0000002" in terms
        assert "GO:0000003" not in terms

    def test_empty_protein_list(self, train_annotations: pd.DataFrame) -> None:
        """Test that empty protein list returns empty DataFrame."""
        baseline = FrequencyBaseline()
        baseline.fit(train_annotations)
        
        predictions = baseline.predict([])
        assert len(predictions) == 0

    def test_get_term_scores_returns_dict(self, train_annotations: pd.DataFrame) -> None:
        """Test that get_term_scores() returns score dictionary."""
        baseline = FrequencyBaseline()
        baseline.fit(train_annotations)
        
        scores = baseline.get_term_scores()
        assert isinstance(scores, dict)
        assert len(scores) == 3
