"""Tests for submission file utilities."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from cafa_6_protein.submission import create_submission, validate_submission


class TestCreateSubmission:
    """Tests for submission file creation."""

    def test_creates_valid_file(self) -> None:
        """Test creating a basic submission file."""
        predictions = pd.DataFrame(
            {
                "protein_id": ["P12345", "P12345", "P67890"],
                "go_term": ["GO:0005524", "GO:0016887", "GO:0005833"],
                "score": [0.95, 0.87, 0.65],
            }
        )

        with NamedTemporaryFile(suffix=".tsv", delete=False) as f:
            output_path = Path(f.name)

        create_submission(predictions, output_path)

        # Read back and validate
        result = pd.read_csv(
            output_path, sep="\t", header=None, names=["protein_id", "go_term", "score"]
        )
        assert len(result) == 3
        assert list(result["protein_id"]) == ["P12345", "P12345", "P67890"]

    def test_limits_terms_per_protein(self) -> None:
        """Test that terms per protein are limited."""
        # Create predictions with many terms for one protein
        predictions = pd.DataFrame(
            {
                "protein_id": ["P12345"] * 10,
                "go_term": [f"GO:000{i:04d}" for i in range(10)],
                "score": [0.9 - i * 0.05 for i in range(10)],
            }
        )

        with NamedTemporaryFile(suffix=".tsv", delete=False) as f:
            output_path = Path(f.name)

        create_submission(predictions, output_path, max_terms_per_protein=5)

        result = pd.read_csv(
            output_path, sep="\t", header=None, names=["protein_id", "go_term", "score"]
        )
        assert len(result) == 5


class TestValidateSubmission:
    """Tests for submission validation."""

    def test_valid_submission(self) -> None:
        """Test that a valid submission passes validation."""
        content = "P12345\tGO:0005524\t0.95\nP67890\tGO:0016887\t0.87\n"

        with NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(content)
            f.flush()

            errors = validate_submission(Path(f.name))
            assert len(errors) == 0

    def test_invalid_score_range(self) -> None:
        """Test detection of invalid score range."""
        content = "P12345\tGO:0005524\t1.5\n"  # Score > 1

        with NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(content)
            f.flush()

            errors = validate_submission(Path(f.name))
            assert any("score" in e.lower() for e in errors)

    def test_invalid_go_format(self) -> None:
        """Test detection of invalid GO term format."""
        content = "P12345\tINVALID\t0.95\n"

        with NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(content)
            f.flush()

            errors = validate_submission(Path(f.name))
            assert any("go term" in e.lower() for e in errors)
